#ifndef R3D_APP_BENCHMARK_H
#define R3D_APP_BENCHMARK_H
#include "render/render.h"
#include <stdlib.h>
/* Grow with observed frames, not the --seconds frame ceiling. */
typedef struct app_frame_sample {
  r3d_frame_stats phases;
  uint64_t cpu_ns;
} app_frame_sample;
/* Explicit selectors: r3d_frame_stats contains non-timing fields and padding.
 */
enum app_timing_field {
  APP_GPU_TOTAL,
  APP_GPU_RAYCAST,
  APP_GPU_BLIT,
  APP_GPU_GUI,
  APP_CPU_WAIT,
  APP_CPU_ACQUIRE,
  APP_CPU_RECORD,
  APP_CPU_SUBMIT,
  APP_CPU_FRAME
};
static inline uint64_t app_sample_timing(const app_frame_sample *s,
                                         size_t field) {
  switch (field) {
  case APP_GPU_TOTAL:
    return s->phases.gpu_ns;
  case APP_GPU_RAYCAST:
    return s->phases.gpu_raycast_ns;
  case APP_GPU_BLIT:
    return s->phases.gpu_blit_ns;
  case APP_GPU_GUI:
    return s->phases.gpu_gui_ns;
  case APP_CPU_WAIT:
    return s->phases.cpu_wait_ns;
  case APP_CPU_ACQUIRE:
    return s->phases.cpu_acquire_ns;
  case APP_CPU_RECORD:
    return s->phases.cpu_record_ns;
  case APP_CPU_SUBMIT:
    return s->phases.cpu_submit_ns;
  case APP_CPU_FRAME:
    return s->cpu_ns;
  default:
    return 0;
  }
}
typedef struct app_samples {
  app_frame_sample *data;
  size_t n, cap;
  bool failed;
} app_samples;
static inline bool app_samples_push(app_samples *s,
                                    const r3d_frame_stats *phases,
                                    uint64_t cpu_ns) {
  if (s->failed)
    return false;
  if (s->n == s->cap) {
    size_t cap = s->cap ? s->cap * 2 : 256;
    if (cap < s->cap || cap > SIZE_MAX / sizeof *s->data) {
      s->failed = true;
      return false;
    }
    app_frame_sample *data = realloc(s->data, cap * sizeof *data);
    if (!data) {
      s->failed = true;
      return false;
    }
    s->data = data;
    s->cap = cap;
  }
  s->data[s->n++] = (app_frame_sample){*phases, cpu_ns};
  return true;
}
static inline r3d_bricks_stats app_bricks_delta(const r3d_bricks_stats *end,
                                                const r3d_bricks_stats *begin) {
  r3d_bricks_stats d = *end;
  d.decoded -= begin->decoded;
  d.jobs -= begin->jobs;
  d.stream_ns -= begin->stream_ns;
  d.failures -= begin->failures;
  d.net_fetched -= begin->net_fetched;
  d.net_encoded -= begin->net_encoded;
  for (int i = 0; i < 8; i++)
    d.lod_requests[i] -= begin->lod_requests[i];
  return d;
}
typedef struct app_bench_intervals {
  r3d_bricks_stats startup, warmup, measured, final_flush;
  uint64_t startup_ns, warmup_ns, measured_ns, final_flush_ns;
} app_bench_intervals;
#endif
