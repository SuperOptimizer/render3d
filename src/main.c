/* render3d — volumetric renderer for Vesuvius Challenge micro-CT volumes.
 * M1: SDL3 window + Vulkan compute raycaster (see spec/ and docs/measured.md). */
#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cimgui.h"
#include "core/camera.h"
#include "core/transfer.h"
#include "core/volume.h"
#include "core/input.h"
#include "core/screenshot.h"
#include "core/stats.h"
#include "render/render.h"
#include "vk/vkctx.h"

#ifndef R3D_SPV_DIR
#define R3D_SPV_DIR "spv" /* release fallback: exe-relative */
#endif

#define MOUSE_SENS 0.0025f
#define ORBIT_SENS 0.006f
#define BASE_SPEED 0.4f /* volume units per second */

enum { CAM_ORBIT = 0, CAM_FLY = 1 };

static void gui_event_hook(void *ud, const SDL_Event *ev) {
  r3d_gui_event((r3d_renderer *)ud, ev);
}

static void take_screenshot(r3d_renderer *renderer, uint64_t frame) {
  uint32_t w = 0, h = 0;
  if (r3d_read_frame(renderer, NULL, &w, &h) != 0) return;
  uint8_t *rgba = malloc((size_t)w * h * 4);
  if (!rgba) return;
  if (r3d_read_frame(renderer, rgba, &w, &h) == 0) {
    char path[64];
    snprintf(path, sizeof path, "render3d_%llu.ppm", (unsigned long long)frame);
    if (r3d_screenshot_ppm(path, rgba, w, h) == 0)
      printf("screenshot: %s (%ux%u)\n", path, w, h);
  }
  free(rgba);
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--probe") == 0) {
    r3d_vkctx vk;
    if (r3d_vkctx_create(&vk, NULL, 0, false) != 0) return EXIT_FAILURE;
    r3d_vkctx_print_caps(&vk);
    r3d_vkctx_destroy(&vk);
    return EXIT_SUCCESS;
  }

  /* automation flags (tests/CI): exit after N frames, dump a screenshot */
  uint32_t exit_frames = 0;
  const char *shot_path = NULL;
  int force_mode = -1, tf_preset = -1;
  int win_w = 1280, win_h = 720;
  float cam0[5] = {0.5f, 0.5f, -1.5f, 0.0f, 0.0f}; /* pos, yaw, pitch */
  bool no_vsync = false;
  float lowcut0 = 0.0f;
  const char *bench = NULL; /* scripted camera path: orbit | zoom | fly */
  for (int i = 1; i < argc; i++) {
    if (i < argc - 1 && strcmp(argv[i], "--frames") == 0) exit_frames = (uint32_t)atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--shot") == 0) shot_path = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--mode") == 0) force_mode = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--tf") == 0) tf_preset = atoi(argv[i + 1]);
    if (i < argc - 2 && strcmp(argv[i], "--size") == 0) {
      win_w = atoi(argv[i + 1]);
      win_h = atoi(argv[i + 2]);
    }
    if (i < argc - 5 && strcmp(argv[i], "--cam") == 0)
      for (int k = 0; k < 5; k++) cam0[k] = (float)atof(argv[i + 1 + k]);
    if (strcmp(argv[i], "--no-vsync") == 0) no_vsync = true;
    if (i < argc - 1 && strcmp(argv[i], "--lowcut") == 0) lowcut0 = (float)atof(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--bench") == 0) bench = argv[i + 1];
  }
  if (bench && exit_frames == 0) exit_frames = 300;

  if (!SDL_Init(SDL_INIT_VIDEO)) {
    fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
    return EXIT_FAILURE;
  }
  SDL_Window *win = SDL_CreateWindow("render3d", win_w, win_h,
                                     SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN);
  if (!win) {
    fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
    SDL_Quit();
    return EXIT_FAILURE;
  }

  r3d_config cfg = {.validate = false, .vsync = !no_vsync, .spv_dir = R3D_SPV_DIR};
  r3d_renderer *renderer = NULL;
  if (r3d_create(win, &cfg, &renderer) != 0) {
    fprintf(stderr, "renderer init failed\n");
    SDL_DestroyWindow(win);
    SDL_Quit();
    return EXIT_FAILURE;
  }

  /* positional args: <volume.u8> <nx> <ny> <nz> */
  uint32_t mode = R3D_MODE_RAYDIR;
  if (argc >= 5 && argv[1][0] != '-') {
    r3d_volume vol;
    if (r3d_volume_open(&vol, argv[1], (uint32_t)atoi(argv[2]), (uint32_t)atoi(argv[3]),
                        (uint32_t)atoi(argv[4])) != 0)
      return EXIT_FAILURE;
    r3d_volume_desc desc = {
        .nx = vol.nx, .ny = vol.ny, .nz = vol.nz, .brick_dim = R3D_BRICK_DIM};
    int up = r3d_upload_volume(renderer, &desc, vol.voxels);
    r3d_volume_close(&vol); /* GPU has it; drop the mapping */
    if (up != 0) return EXIT_FAILURE;
    mode = R3D_MODE_FULL;
  }
  if (force_mode >= 0) mode = (uint32_t)force_mode % R3D_MODE_COUNT;
  if (tf_preset >= 0) {
    r3d_tf tf;
    uint8_t lut[256][4];
    r3d_tf_preset((uint32_t)tf_preset, &tf);
    r3d_tf_build(&tf, lut);
    r3d_set_transfer(renderer, lut);
  }

  /* orbit (turntable around the volume) is the default; --cam implies fly */
  bool cam_given = false;
  for (int i = 1; i < argc; i++)
    if (strcmp(argv[i], "--cam") == 0) cam_given = true;
  int cam_mode = cam_given ? CAM_FLY : CAM_ORBIT;

  r3d_camera cam;
  r3d_camera_init(&cam, v3(cam0[0], cam0[1], cam0[2]));
  cam.yaw = cam0[3];
  cam.pitch = cam0[4];
  if (cam_mode == CAM_ORBIT) r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, 0.5f), 2.0f);
  r3d_input in = {0};
  r3d_stats stats;
  r3d_stats_init(&stats);

  float step_voxels = 1.0f, density = 1.0f, lod_bias = 0.0f;
  float low_cut = lowcut0; /* voxel-value threshold, 0..255 */
  uint32_t tf_idx = tf_preset > 0 ? (uint32_t)tf_preset : 0;
  uint32_t frame_index = 0;
  float fps_smooth = 60.0f;
  uint64_t last_gpu_ns = 0;
  r3d_frame_stats prof = {0};   /* EMA-smoothed for display */
  r3d_frame_stats prof_sum = {0}; /* running sums for the exit report */
  uint64_t prof_frames = 0;
  uint64_t prev_ns = r3d_now_ns();

  bool running = true;
  while (running) {
    uint64_t t0 = r3d_now_ns();
    float dt = (float)((double)(t0 - prev_ns) / 1e9);
    prev_ns = t0;
    if (dt > 0.1f) dt = 0.1f;

    ImGuiIO *io = igGetIO_Nil(); /* Want* flags reflect last frame — fine */
    r3d_input_poll(&in, win, gui_event_hook, renderer, !io->WantCaptureMouse,
                   cam_mode == CAM_FLY);
    if (io->WantCaptureKeyboard && !in.captured)
      in.move[0] = in.move[1] = in.move[2] = 0.0f;
    if (in.quit) running = false;
    if (in.resized) r3d_resize(renderer);
    if (in.mode_delta) {
      mode = (mode + (uint32_t)in.mode_delta) % R3D_MODE_COUNT;
      printf("mode: %u\n", mode);
    }
    if (in.tf_delta) {
      tf_idx = (tf_idx + 1) % r3d_tf_preset(UINT32_MAX, NULL);
      r3d_tf tf;
      uint8_t lut[256][4];
      r3d_tf_preset(tf_idx, &tf);
      r3d_tf_build(&tf, lut);
      r3d_set_transfer(renderer, lut);
      printf("tf preset: %u\n", tf_idx);
    }
    step_voxels *= in.step_scale;
    density *= in.density_scale;
    lod_bias += in.lod_delta;
    if (cam_mode == CAM_ORBIT) {
      /* drag "grabs" the cube: drag right spins it right, wheel zooms, WASD pans */
      if (in.dragging)
        r3d_camera_orbit_drag(&cam, -in.look[0] * ORBIT_SENS, -in.look[1] * ORBIT_SENS);
      if (in.wheel != 0.0f && !io->WantCaptureMouse)
        r3d_camera_orbit_zoom(&cam, powf(0.9f, in.wheel));
      float pan = BASE_SPEED * cam.dist * (in.fast ? 5.0f : 1.0f);
      if (in.move[0] != 0.0f || in.move[1] != 0.0f || in.move[2] != 0.0f)
        r3d_camera_orbit_pan(&cam, v3(in.move[0], in.move[1], in.move[2]), pan * dt);
    } else {
      r3d_camera_look(&cam, in.look[0] * MOUSE_SENS, -in.look[1] * MOUSE_SENS);
      float speed = BASE_SPEED * (in.fast ? 5.0f : 1.0f);
      r3d_camera_move(&cam, v3(in.move[0], in.move[1], in.move[2]), speed * dt);
    }

    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(win, &w, &h);
    if (w <= 0 || h <= 0) {
      SDL_Delay(50);
      continue;
    }

    /* scripted camera paths for reproducible perf runs (override user input) */
    if (bench) {
      float ph = (float)frame_index / (float)exit_frames; /* 0..1 over the run */
      float tau = 6.2831853f;
      if (strcmp(bench, "orbit") == 0) {
        cam.yaw = ph * tau;
        cam.pitch = 0.5f * sinf(ph * tau * 2.0f);
        r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, 0.5f), 2.0f);
      } else if (strcmp(bench, "zoom") == 0) {
        cam.yaw = ph * tau * 0.5f;
        cam.pitch = 0.2f;
        r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, 0.5f),
                             1.75f - 1.55f * sinf(ph * 3.14159265f));
      } else { /* fly: weaving pass straight through the volume */
        cam.pos = v3(0.5f + 0.15f * sinf(ph * tau * 1.5f), 0.5f + 0.1f * sinf(ph * tau),
                     -0.3f + 1.6f * ph);
        cam.yaw = 0.15f * sinf(ph * tau);
        cam.pitch = 0.1f * cosf(ph * tau);
      }
    }
    r3d_v3 right, up, fwd;
    r3d_camera_basis(&cam, (float)w / (float)h, &right, &up, &fwd);

    /* control panel */
    fps_smooth = fps_smooth * 0.95f + (dt > 0 ? 0.05f / dt : 0.0f);
    r3d_gui_begin(renderer);
    igSetNextWindowPos((ImVec2){10, 10}, ImGuiCond_FirstUseEver, (ImVec2){0, 0});
    igBegin("render3d", NULL, ImGuiWindowFlags_AlwaysAutoResize);
    igText("%.0f fps   gpu %.2f ms", (double)fps_smooth, (double)last_gpu_ns / 1e6);
    int m = (int)mode;
    if (igCombo_Str("mode", &m, "full\0mip\0depth\0heatmap\0raydir\0flat\0", 6))
      mode = (uint32_t)m;
    int prev_cm = cam_mode;
    igCombo_Str("camera", &cam_mode, "orbit\0fly\0", 2);
    if (cam_mode == CAM_ORBIT && prev_cm == CAM_FLY)
      r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, 0.5f), 2.0f);
    int t = (int)tf_idx;
    if (igCombo_Str("transfer fn", &t, "gray\0scroll\0high-pass\0", 3)) {
      tf_idx = (uint32_t)t;
      r3d_tf tfp;
      uint8_t lut[256][4];
      r3d_tf_preset(tf_idx, &tfp);
      r3d_tf_build(&tfp, lut);
      r3d_set_transfer(renderer, lut);
    }
    igSliderFloat("step (voxels)", &step_voxels, 0.25f, 4.0f, "%.2f", 0);
    igSliderFloat("density", &density, 0.1f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    igSliderFloat("low cut", &low_cut, 0.0f, 255.0f, "%.0f", 0);
    igSliderFloat("lod bias", &lod_bias, -2.0f, 4.0f, "%.2f", 0);
    igText("cam (%.2f %.2f %.2f) yaw %.2f pitch %.2f", (double)cam.pos.x, (double)cam.pos.y,
           (double)cam.pos.z, (double)cam.yaw, (double)cam.pitch);
    if (igCollapsingHeader_TreeNodeFlags("profile", 0)) {
      igText("gpu total   %6.2f ms", (double)prof.gpu_ns / 1e6);
      igText("  raycast   %6.2f ms", (double)prof.gpu_raycast_ns / 1e6);
      igText("  blit      %6.2f ms", (double)prof.gpu_blit_ns / 1e6);
      igText("  gui       %6.2f ms", (double)prof.gpu_gui_ns / 1e6);
      igText("cpu wait    %6.2f ms", (double)prof.cpu_wait_ns / 1e6);
      igText("cpu acquire %6.2f ms", (double)prof.cpu_acquire_ns / 1e6);
      igText("cpu record  %6.2f ms", (double)prof.cpu_record_ns / 1e6);
      igText("cpu submit  %6.2f ms", (double)prof.cpu_submit_ns / 1e6);
    }
    if (cam_mode == CAM_ORBIT)
      igTextDisabled("drag: rotate   wheel: zoom   WASD: pan   F12: shot");
    else
      igTextDisabled("click: fly (Esc releases)   WASD+QE: move   F12: shot");
    igEnd();

    r3d_frame_params p = {
        .cam_origin = {cam.pos.x, cam.pos.y, cam.pos.z},
        .cam_right = {right.x, right.y, right.z},
        .cam_up = {up.x, up.y, up.z},
        .cam_forward = {fwd.x, fwd.y, fwd.z},
        .step_voxels = step_voxels,
        .density = density,
        .lod_bias = lod_bias,
        .max_mip = 10.0f,
        .viewport = {(uint32_t)w, (uint32_t)h},
        .mode = mode,
        .frame_index = frame_index++,
        .threshold = low_cut / 255.0f,
    };
    r3d_frame_stats st = {0};
    int frc = r3d_frame(renderer, &p, &st);
    if (frc == 0) {
      last_gpu_ns = st.gpu_ns;
      const uint64_t *sv = (const uint64_t *)&st;
      uint64_t *pv = (uint64_t *)&prof, *qv = (uint64_t *)&prof_sum;
      for (size_t k = 0; k < sizeof st / sizeof(uint64_t); k++) {
        pv[k] = (uint64_t)((double)pv[k] * 0.95 + (double)sv[k] * 0.05);
        qv[k] += sv[k];
      }
      prof_frames++;
    }
    if (frc < 0) {
      fprintf(stderr, "r3d_frame failed\n");
      running = false;
    }
    if (in.screenshot) take_screenshot(renderer, stats.frame_index);
    if (exit_frames && frame_index >= exit_frames) {
      if (shot_path) {
        uint32_t sw = 0, sh = 0;
        r3d_read_frame(renderer, NULL, &sw, &sh);
        uint8_t *rgba = malloc((size_t)sw * sh * 4);
        if (rgba && r3d_read_frame(renderer, rgba, &sw, &sh) == 0)
          r3d_screenshot_ppm(shot_path, rgba, sw, sh);
        free(rgba);
      }
      running = false;
    }

    r3d_stats_push(&stats, r3d_now_ns() - t0, st.gpu_ns);
    r3d_stats_report(&stats);
  }

  r3d_stats_report_now(&stats);
  if (prof_frames > 2) {
    /* skip warmup skew: averages include first frames with empty queries */
    double n = (double)prof_frames;
    printf("profile avg: gpu %.2f (raycast %.2f blit %.2f gui %.2f) | "
           "wait %.2f acquire %.2f record %.2f submit %.2f ms\n",
           (double)prof_sum.gpu_ns / n / 1e6, (double)prof_sum.gpu_raycast_ns / n / 1e6,
           (double)prof_sum.gpu_blit_ns / n / 1e6, (double)prof_sum.gpu_gui_ns / n / 1e6,
           (double)prof_sum.cpu_wait_ns / n / 1e6, (double)prof_sum.cpu_acquire_ns / n / 1e6,
           (double)prof_sum.cpu_record_ns / n / 1e6, (double)prof_sum.cpu_submit_ns / n / 1e6);
  }
  r3d_destroy(renderer);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return EXIT_SUCCESS;
}
