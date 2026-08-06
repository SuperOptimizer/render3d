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

/* first voxel value with nonzero TF alpha: below it, samples are invisible */
static float tf_min_visible(const uint8_t lut[256][4]) {
  for (uint32_t i = 0; i < 256; i++)
    if (lut[i][3] != 0) return (float)i;
  return 255.0f;
}

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
  float volpos0[3] = {0, 0, 0}, volrot0[3] = {0, 0, 0};
  uint32_t slab_wz = 0;     /* nonzero = slab mode, max visible depth */
  int depth0 = 0;           /* initial visible depth (default = max) */
  bool clip_mode = false;   /* clipmap over the shard band */
  const char *bricks_path = NULL; /* c5d shard for GPU-decoded bricks mode */
  int pool_bpa = 0, warm_mb = 0;  /* bricks hot-atlas slots/axis, warm-tier MB */
  bool vslab_mode = false;        /* toroidal streaming window over the export */
  int vsw = 12096, vsh = 12096, vsd = 16; /* window dims (voxels) */
  long long vsz0 = 34288;         /* start z (world; default inside the local band) */
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
    if (i < argc - 3 && strcmp(argv[i], "--volpos") == 0)
      for (int k = 0; k < 3; k++) volpos0[k] = (float)atof(argv[i + 1 + k]);
    if (i < argc - 3 && strcmp(argv[i], "--volrot") == 0)
      for (int k = 0; k < 3; k++) volrot0[k] = (float)atof(argv[i + 1 + k]) / 57.29578f;
    if (i < argc - 1 && strcmp(argv[i], "--depth") == 0) depth0 = atoi(argv[i + 1]);
    if (strcmp(argv[i], "--clipmap") == 0) clip_mode = true;
    if (i < argc - 1 && strcmp(argv[i], "--bricks") == 0) bricks_path = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--pool") == 0) pool_bpa = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--warm") == 0) warm_mb = atoi(argv[i + 1]);
    if (strcmp(argv[i], "--vslab") == 0) vslab_mode = true;
    if (i < argc - 1 && strcmp(argv[i], "--vsz") == 0) vsz0 = atoll(argv[i + 1]);
    if (i < argc - 3 && strcmp(argv[i], "--vswin") == 0) {
      vsw = atoi(argv[i + 1]);
      vsh = atoi(argv[i + 2]);
      vsd = atoi(argv[i + 3]);
    }
    if (strcmp(argv[i], "--slab") == 0) {
      slab_wz = 32;
      if (i < argc - 1 && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
        slab_wz = (uint32_t)atoi(argv[i + 1]);
    }
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
  r3d_volume slab_src = {0}; /* stays open in slab mode (window scrolls it) */
  uint32_t slab_z0 = 0, slab_z0_max = 0;
  if (argc >= 5 && argv[1][0] != '-') {
    r3d_volume vol;
    if (r3d_volume_open(&vol, argv[1], (uint32_t)atoi(argv[2]), (uint32_t)atoi(argv[3]),
                        (uint32_t)atoi(argv[4])) != 0)
      return EXIT_FAILURE;
    if (slab_wz) {
      /* --slab N = max visible depth; ring holds N+2 so face filtering never
       * crosses the seam and depth changes need no re-upload */
      uint32_t ring = slab_wz + 2;
      r3d_slab_desc sd = {.nx = vol.nx, .ny = vol.ny, .nz = vol.nz, .wz = ring};
      if (r3d_slab_init(renderer, &sd) != 0) return EXIT_FAILURE;
      slab_src = vol;
      slab_z0 = (vol.nz - ring) / 2; /* start mid-depth */
      slab_z0_max = vol.nz - ring;
      if (r3d_slab_window(renderer, &slab_src, slab_z0) != 0) return EXIT_FAILURE;
    } else {
      r3d_volume_desc desc = {
          .nx = vol.nx, .ny = vol.ny, .nz = vol.nz, .brick_dim = R3D_BRICK_DIM};
      int up = r3d_upload_volume(renderer, &desc, vol.voxels);
      r3d_volume_close(&vol); /* GPU has it; drop the mapping */
      if (up != 0) return EXIT_FAILURE;
    }
    mode = R3D_MODE_FULL;
  } else if (slab_wz) {
    fprintf(stderr, "--slab needs a volume argument\n");
    return EXIT_FAILURE;
  }

  if (bricks_path) {
    if (r3d_bricks_begin(renderer, bricks_path, (uint32_t)pool_bpa, (uint32_t)warm_mb) != 0)
      return EXIT_FAILURE;
    mode = R3D_MODE_FULL;
  }

  int64_t vs_z0 = (int64_t)vsz0;
  double vs_fx = 21504.0, vs_fy = 21504.0;
  bool vs_follow = true;
  uint64_t vs_pend_acc = 0;
  if (vslab_mode) {
    if (r3d_vslab_begin(renderer, "band", (uint32_t)vsw, (uint32_t)vsh, (uint32_t)vsd) != 0)
      return EXIT_FAILURE;
    mode = R3D_MODE_FULL;
  }

  /* clipmap: 43k^2 cross sections from the shard band + pyramid */
  const uint32_t CLIP_NX = 43008, CLIP_BAND_Z = 33, CLIP_DEPTH_MAX = 32;
  uint64_t clip_z0 = 0, clip_z0_min = 0, clip_z0_max = 0;
  if (clip_mode) {
    if (r3d_clip_begin(renderer, "band", "pyramid", CLIP_BAND_Z, CLIP_DEPTH_MAX) != 0)
      return EXIT_FAILURE;
    clip_z0_min = (uint64_t)CLIP_BAND_Z * 1024;
    clip_z0_max = clip_z0_min + 1024 - CLIP_DEPTH_MAX;
    clip_z0 = clip_z0_min + 512 - CLIP_DEPTH_MAX / 2;
    mode = R3D_MODE_FULL;
  }
  if (force_mode >= 0) mode = (uint32_t)force_mode % R3D_MODE_COUNT;
  float tf_min_v = 1.0f; /* backend default ramp: alpha nonzero from value 1 */
  if (tf_preset >= 0) {
    r3d_tf tf;
    uint8_t lut[256][4];
    r3d_tf_preset((uint32_t)tf_preset, &tf);
    r3d_tf_build(&tf, lut);
    r3d_set_transfer(renderer, lut);
    tf_min_v = tf_min_visible(lut);
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
  if (cam_mode == CAM_ORBIT) {
    if (clip_mode) {
      float ez = (float)CLIP_DEPTH_MAX / (float)CLIP_NX;
      r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, ez * 0.5f), 1.3f);
    } else if (vslab_mode) {
      float ey = (float)vsh / (float)vsw;
      float ez = (float)(vsd + 2) / (float)vsw;
      r3d_camera_orbit_set(&cam, v3(0.5f, ey * 0.5f, ez * 0.5f), 1.4f);
    } else if (slab_wz) {
      /* orbit the thin slab: target its center, sit back along z */
      float ey = (float)slab_src.ny / (float)slab_src.nx;
      float ez = (float)slab_wz / (float)slab_src.nx;
      r3d_camera_orbit_set(&cam, v3(0.5f, ey * 0.5f, ez * 0.5f), 1.4f);
    } else {
      r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, 0.5f), 2.0f);
    }
  }
  r3d_input in = {0};
  r3d_stats stats;
  r3d_stats_init(&stats);

  float step_voxels = 1.0f, density = 1.0f, lod_bias = 0.0f;
  float low_cut = lowcut0; /* voxel-value threshold, 0..255 */
  bool auto_scroll = false;
  float auto_speed = 10.0f; /* slices per second */
  float auto_accum = 0.0f;
  int slab_depth = depth0 >= 2 && depth0 <= (int)slab_wz ? depth0 : (int)slab_wz;
  int clip_depth = depth0 >= 2 && depth0 <= (int)CLIP_DEPTH_MAX ? depth0 : (int)CLIP_DEPTH_MAX;
  uint32_t clip_valid_disp = 0;

  /* volume (model) transform: translation in world units, rotation ypr */
  r3d_v3 vol_t = v3(volpos0[0], volpos0[1], volpos0[2]);
  float vol_rot[3] = {volrot0[0], volrot0[1], volrot0[2]}; /* yaw, pitch, roll (radians) */
  float fov_deg = 60.0f;
  uint32_t tf_idx = tf_preset > 0 ? (uint32_t)tf_preset : 0;
  uint32_t frame_index = 0;
  float fps_smooth = 60.0f;
  bool adaptive_res = true; /* half-res rendering while the camera moves */
  int settle = 0;
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
    if (clip_mode) {
      int64_t nz0 = (int64_t)clip_z0 + in.zdelta + (int64_t)in.zpage * (int64_t)clip_depth;
      if (nz0 < (int64_t)clip_z0_min) nz0 = (int64_t)clip_z0_min;
      if (nz0 > (int64_t)clip_z0_max) nz0 = (int64_t)clip_z0_max;
      clip_z0 = (uint64_t)nz0;
    }
    if (slab_wz) {
      int64_t nz0 = (int64_t)slab_z0 + in.zdelta + (int64_t)in.zpage * (int64_t)slab_depth;
      if (bench && strcmp(bench, "zsweep") == 0) /* scripted scroll for perf/tests */
        nz0 = (int64_t)((float)slab_z0_max * (float)frame_index / (float)exit_frames);
      if (auto_scroll) {
        auto_accum += auto_speed * dt;
        float whole = floorf(auto_accum);
        nz0 += (int64_t)whole;
        auto_accum -= whole;
        if (nz0 >= (int64_t)slab_z0_max) { /* bounce at the ends */
          nz0 = slab_z0_max;
          auto_speed = -auto_speed;
        } else if (nz0 <= 0 && auto_speed < 0) {
          nz0 = 0;
          auto_speed = -auto_speed;
        }
      }
      if (nz0 < 0) nz0 = 0;
      if (nz0 > (int64_t)slab_z0_max) nz0 = slab_z0_max;
      if ((uint32_t)nz0 != slab_z0) {
        slab_z0 = (uint32_t)nz0;
        r3d_slab_window(renderer, &slab_src, slab_z0);
      }
    }
    if (in.tf_delta) {
      tf_idx = (tf_idx + 1) % r3d_tf_preset(UINT32_MAX, NULL);
      r3d_tf tf;
      uint8_t lut[256][4];
      r3d_tf_preset(tf_idx, &tf);
      r3d_tf_build(&tf, lut);
      r3d_set_transfer(renderer, lut);
      tf_min_v = tf_min_visible(lut);
      printf("tf preset: %u\n", tf_idx);
    }
    step_voxels *= in.step_scale;
    density *= in.density_scale;
    lod_bias += in.lod_delta;
    cam.fov_y = fov_deg * 0.01745329f;
    if (cam_mode == CAM_ORBIT) {
      /* drag grabs the cube; Shift pans the camera; Ctrl translates the
       * volume; Ctrl+Shift rotates the volume */
      if (in.dragging && (in.ctrl || in.fast)) {
        r3d_v3 br, bu, bf;
        r3d_camera_basis(&cam, 1.0f, &br, &bu, &bf);
        r3d_v3 ru = v3_norm(br), uu = v3_norm(bu);
        float k = cam.dist * 0.0012f;
        if (in.ctrl && in.fast) { /* rotate volume */
          vol_rot[0] += in.look[0] * ORBIT_SENS;
          vol_rot[1] += in.look[1] * ORBIT_SENS;
        } else if (in.ctrl) { /* translate volume in the view plane */
          vol_t = v3_add(vol_t, v3_add(v3_scale(ru, in.look[0] * k),
                                       v3_scale(uu, -in.look[1] * k)));
        } else { /* pan camera (grab-the-world) */
          r3d_camera_orbit_pan(&cam, v3(-in.look[0], in.look[1], 0), k);
        }
      } else if (in.dragging) {
        r3d_camera_orbit_drag(&cam, -in.look[0] * ORBIT_SENS, -in.look[1] * ORBIT_SENS);
      }
      if (in.wheel != 0.0f && !io->WantCaptureMouse)
        /* shift+wheel zooms 3x faster: whole-scroll to fiber scale is a ~3200x
         * distance ratio, a long ride at 0.9/detent */
        r3d_camera_orbit_zoom(&cam, powf(in.fast ? 0.73f : 0.9f, in.wheel));
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
      } else if (strcmp(bench, "fly") == 0) { /* weaving pass through the volume */
        cam.pos = v3(0.5f + 0.15f * sinf(ph * tau * 1.5f), 0.5f + 0.1f * sinf(ph * tau),
                     -0.3f + 1.6f * ph);
        cam.yaw = 0.15f * sinf(ph * tau);
        cam.pitch = 0.1f * cosf(ph * tau);
      } else if (strcmp(bench, "clippan") == 0) { /* sweep across the cross-section */
        cam.pos = v3(0.25f + 0.5f * ph, 0.5f, -0.02f);
        cam.yaw = 0.0f;
        cam.pitch = 0.0f;
      } else if (strcmp(bench, "zoomio") == 0) {
        /* full zoom sweep: whole-composite view down to voxel scale and back
         * (log-space triangle wave) — walks every LOD band the mode has */
        float ey = slab_wz ? (float)slab_src.ny / (float)slab_src.nx : 1.0f;
        float ez = slab_wz ? (float)slab_wz / (float)slab_src.nx : 1.0f;
        if (vslab_mode) { ey = (float)vsh / (float)vsw; ez = (float)(vsd + 2) / (float)vsw; }
        float tri = ph < 0.5f ? ph * 2.0f : 2.0f - ph * 2.0f; /* 0->1->0 */
        float d = 1.6f * powf(0.002f / 1.6f, tri);
        cam.yaw = 0.0f;
        cam.pitch = 0.0f;
        r3d_camera_orbit_set(&cam, v3(0.5f, ey * 0.5f, ez * 0.5f), d);
      } else if (strcmp(bench, "volrot") == 0) {
        /* mid-zoom + model rotation: worst case for anisotropic footprints */
        float ey = slab_wz ? (float)slab_src.ny / (float)slab_src.nx : 1.0f;
        float ez = slab_wz ? (float)slab_wz / (float)slab_src.nx : 1.0f;
        if (vslab_mode) { ey = (float)vsh / (float)vsw; ez = (float)(vsd + 2) / (float)vsw; }
        vol_rot[0] = 0.7f * sinf(ph * tau);
        vol_rot[1] = 0.45f * sinf(ph * tau * 0.7f);
        cam.yaw = 0.0f;
        cam.pitch = 0.0f;
        r3d_camera_orbit_set(&cam, v3(0.5f, ey * 0.5f, ez * 0.5f), 0.2f);
      } /* other bench names (zsweep) keep the default camera */
      if (vslab_mode && strcmp(bench, "zsweep") == 0) /* scroll z across the band */
        vs_z0 = 33792 + (int64_t)(ph * (1024.0f - (float)(vsd + 2)));
      if (vslab_mode && strcmp(bench, "clippan") == 0) { /* xy window churn */
        vs_follow = false;
        vs_fx = 12000.0 + (double)ph * 18000.0;
        vs_fy = 21504.0;
      }
    }
    r3d_v3 right, up, fwd;
    r3d_camera_basis(&cam, (float)w / (float)h, &right, &up, &fwd);

    /* adaptive resolution: drop to half res while interacting (4x fewer
     * rays), snap back to full once the camera settles */
    bool moving = in.dragging || in.captured || in.wheel != 0.0f || in.zdelta || in.zpage ||
                  auto_scroll ||
                  in.move[0] != 0.0f || in.move[1] != 0.0f || in.move[2] != 0.0f;
    settle = moving ? 15 : (settle > 0 ? settle - 1 : 0);
    bool half_res = adaptive_res && settle > 0;
    if (getenv("R3D_FORCE_HALF")) half_res = true; /* testing/benching the path */
    else if (in.screenshot || (exit_frames && shot_path && frame_index + 1 >= exit_frames))
      half_res = false; /* captures always full res */
    uint32_t rvw = half_res ? (uint32_t)w / 2 : (uint32_t)w;
    uint32_t rvh = half_res ? (uint32_t)h / 2 : (uint32_t)h;

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
      tf_min_v = tf_min_visible(lut);
    }
    igSliderFloat("step (voxels)", &step_voxels, 0.25f, 4.0f, "%.2f", 0);
    igSliderFloat("density", &density, 0.1f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    igSliderFloat("low cut", &low_cut, 0.0f, 255.0f, "%.0f", 0);
    if (slab_wz) {
      igSliderInt("depth (voxels)", &slab_depth, 2, (int)slab_wz, "%d", 0);
      int z0i = (int)slab_z0;
      if (igSliderInt("z position", &z0i, 0, (int)slab_z0_max, "%d", 0)) {
        slab_z0 = (uint32_t)z0i;
        r3d_slab_window(renderer, &slab_src, slab_z0);
      }
      igCheckbox("auto-scroll", &auto_scroll);
      igSameLine(0, 10);
      igSliderFloat("slices/s", &auto_speed, -60.0f, 60.0f, "%.0f", 0);
    }
    if (clip_mode) {
      igSliderInt("depth (voxels)", &clip_depth, 2, (int)CLIP_DEPTH_MAX, "%d", 0);
      int z0i = (int)(clip_z0 - clip_z0_min);
      if (igSliderInt("z position", &z0i, 0, (int)(clip_z0_max - clip_z0_min), "%d", 0))
        clip_z0 = clip_z0_min + (uint64_t)z0i;
      igText("levels ready:%s%s%s%s%s%s", (clip_valid_disp & 1) ? " L0" : "",
             (clip_valid_disp & 2) ? " L1" : "", (clip_valid_disp & 4) ? " L2" : "",
             (clip_valid_disp & 8) ? " L3" : "", (clip_valid_disp & 16) ? " L4" : "",
             (clip_valid_disp & 32) ? " L5" : "");
    }
    if (vslab_mode) {
      int z0i = (int)vs_z0;
      if (igSliderInt("z position (world)", &z0i, 0, 68608 - (vsd + 2), "%d", 0))
        vs_z0 = z0i;
      igCheckbox("follow camera (x/y)", &vs_follow);
      int64_t vo_[3];
      uint32_t pend;
      r3d_vslab_get(renderer, vo_, &pend);
      igText("window @ (%lld, %lld, %lld)%s", (long long)vo_[0], (long long)vo_[1],
             (long long)vo_[2], pend ? "  streaming..." : "");
    }
    if (bricks_path) {
      r3d_bricks_stats bst;
      r3d_bricks_get_stats(renderer, &bst);
      igText("bricks: hot %u/%u slots  warm %u (%.0f/%llu MB)%s", bst.hot, bst.hot_cap,
             bst.warm_bricks, (double)bst.warm_bytes / 1048576.0,
             (unsigned long long)(bst.warm_cap >> 20), bst.inflight ? "  streaming..." : "");
    }
    igSliderFloat("lod bias", &lod_bias, -2.0f, 4.0f, "%.2f", 0);
    igCheckbox("half-res while moving", &adaptive_res);
    if (igCollapsingHeader_TreeNodeFlags("transform", 0)) {
      float vt[3] = {vol_t.x, vol_t.y, vol_t.z};
      if (igDragFloat3("volume pos", vt, 0.002f, -4.0f, 4.0f, "%.3f", 0))
        vol_t = v3(vt[0], vt[1], vt[2]);
      float vr[3] = {vol_rot[0] * 57.29578f, vol_rot[1] * 57.29578f, vol_rot[2] * 57.29578f};
      if (igDragFloat3("volume rot (ypr)", vr, 0.5f, -180.0f, 180.0f, "%.1f", 0))
        for (int k = 0; k < 3; k++) vol_rot[k] = vr[k] / 57.29578f;
      if (igButton("reset volume", (ImVec2){0, 0})) {
        vol_t = v3(0, 0, 0);
        vol_rot[0] = vol_rot[1] = vol_rot[2] = 0.0f;
      }
      igSeparator();
      float ct[3] = {cam.target.x, cam.target.y, cam.target.z};
      if (igDragFloat3("cam target", ct, 0.002f, -4.0f, 4.0f, "%.3f", 0) &&
          cam_mode == CAM_ORBIT)
        r3d_camera_orbit_set(&cam, v3(ct[0], ct[1], ct[2]), cam.dist);
      if (igDragFloat("cam dist", &cam.dist, 0.005f, 0.0005f, 20.0f, "%.4f", 0) &&
          cam_mode == CAM_ORBIT)
        r3d_camera_orbit_set(&cam, cam.target, cam.dist);
      igSliderFloat("fov", &fov_deg, 20.0f, 120.0f, "%.0f°", 0);
    }
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
      igTextDisabled("drag orbit | shift+drag pan cam | ctrl+drag move vol\n"
                     "ctrl+shift+drag rot vol | wheel zoom | WASD pan | F12 shot");
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
        .viewport = {rvw, rvh},
        .mode = mode,
        .frame_index = frame_index++,
        .threshold = low_cut / 255.0f,
        .skip_gate = fmaxf(low_cut, tf_min_v - 0.5f) / 255.0f,
    };
    r3d_m3 vm = m3_ypr(vol_rot[0], vol_rot[1], vol_rot[2]);
    memcpy(p.vol_r0, &vm.r0, 12);
    memcpy(p.vol_r1, &vm.r1, 12);
    memcpy(p.vol_r2, &vm.r2, 12);
    p.vol_tx = vol_t.x;
    p.vol_ty = vol_t.y;
    p.vol_tz = vol_t.z;
    if (slab_wz) {
      r3d_slab_params(renderer, &p);
      p.slab_depth = (uint32_t)slab_depth;
    }
    if (vslab_mode) {
      /* focus = view axis ^ window mid-plane, in WINDOW space -> world */
      float vey = (float)vsh / (float)vsw, vez = (float)(vsd + 2) / (float)vsw;
      r3d_v3 vc = v3(0.5f, vey * 0.5f, vez * 0.5f);
      r3d_v3 vo = v3_add(m3_tmul(vm, v3_sub(v3_sub(cam.pos, vol_t), vc)), vc);
      r3d_v3 vd = m3_tmul(vm, fwd);
      float tt = vd.z != 0.0f ? (vez * 0.5f - vo.z) / vd.z : 0.0f;
      if (tt < 0.0f) tt = 0.0f;
      float fxn = fclampf(vo.x + vd.x * tt, 0.0f, 1.0f);
      float fyn = fclampf((vo.y + vd.y * tt) / (vey > 0.0f ? vey : 1.0f), 0.0f, 1.0f);
      int64_t vo3[3];
      r3d_vslab_get(renderer, vo3, NULL);
      if (vs_follow && vo3[0] >= 0) {
        vs_fx = (double)vo3[0] + (double)fxn * vsw;
        vs_fy = (double)vo3[1] + (double)fyn * vsh;
      }
      r3d_vslab_frame(renderer, vs_fx, vs_fy, vs_z0, moving ? 1u : 3u, &p);
      /* residency-lag metric: cells short of full residency, second half of
       * the run only (the first half absorbs the initial window fill) */
      if (bench && frame_index * 2 >= exit_frames) {
        int64_t bo_[3];
        uint32_t pd = 0;
        r3d_vslab_get(renderer, bo_, &pd);
        vs_pend_acc += pd;
      }
    }
    if (bricks_path) {
      /* streaming pump: camera in VOLUME space (model transform inverted, like
       * the clip focus); smaller decode budget while moving so the pump's GPU
       * time shares the frame with half-res rendering */
      r3d_v3 vc = v3(0.5f, 0.5f, 0.5f);
      r3d_v3 vo = v3_add(m3_tmul(vm, v3_sub(v3_sub(cam.pos, vol_t), vc)), vc);
      r3d_v3 vd = m3_tmul(vm, fwd);
      float be[3] = {vo.x, vo.y, vo.z}, bf[3] = {vd.x, vd.y, vd.z};
      float asp = (float)w / (float)h;
      float ht = tanf(cam.fov_y * 0.5f) * sqrtf(1.0f + asp * asp);
      r3d_bricks_stream(renderer, be, bf, ht, p.skip_gate, moving ? 2u : 6u);
      r3d_bricks_params(renderer, &p);
    }
    if (clip_mode) {
      /* focus = where the view axis crosses the slab plane, computed in
       * VOLUME space so it tracks the model transform */
      float ezc = (float)CLIP_DEPTH_MAX / (float)CLIP_NX * 0.5f;
      r3d_v3 vc = v3(0.5f, 0.5f, ezc);
      r3d_v3 vo = v3_add(m3_tmul(vm, v3_sub(v3_sub(cam.pos, vol_t), vc)), vc);
      r3d_v3 vd = m3_tmul(vm, fwd);
      float tt = vd.z != 0.0f ? (ezc - vo.z) / vd.z : 0.0f;
      if (tt < 0.0f) tt = 0.0f;
      double fx = (double)(vo.x + vd.x * tt) * CLIP_NX;
      double fy = (double)(vo.y + vd.y * tt) * CLIP_NX;
      if (fx < 0) fx = 0;
      if (fy < 0) fy = 0;
      if (fx > CLIP_NX) fx = CLIP_NX;
      if (fy > CLIP_NX) fy = CLIP_NX;
      p.slab_depth = (uint32_t)clip_depth;
      if (r3d_clip_frame(renderer, fx, fy, clip_z0, &p) != 0) running = false;
      if (p.clip_valid != clip_valid_disp)
        printf("clip: valid=0x%02x z0=%llu focus=(%.0f,%.0f)\n", p.clip_valid,
               (unsigned long long)clip_z0, fx, fy);
      clip_valid_disp = p.clip_valid;
    }
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
  if (slab_src.voxels) r3d_volume_close(&slab_src);
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
  if (vslab_mode && bench)
    printf("vslab bench: pending cell-frames %llu\n", (unsigned long long)vs_pend_acc);
  r3d_destroy(renderer);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return EXIT_SUCCESS;
}
