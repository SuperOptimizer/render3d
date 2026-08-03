#include "core/stats.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

uint64_t r3d_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void r3d_stats_init(r3d_stats *s) {
  memset(s, 0, sizeof *s);
  s->last_report_ns = r3d_now_ns();
}

void r3d_stats_push(r3d_stats *s, uint64_t cpu_ns, uint64_t gpu_ns) {
  s->cpu_ns[s->head] = cpu_ns;
  s->gpu_ns[s->head] = gpu_ns;
  s->head = (s->head + 1u) % R3D_STATS_RING;
  if (s->count < R3D_STATS_RING) s->count++;
  s->frame_index++;
}

void r3d_stats_report(r3d_stats *s) {
  uint64_t now = r3d_now_ns();
  if (now - s->last_report_ns < 1000000000ull || s->count == 0) return;
  s->last_report_ns = now;
  r3d_stats_report_now(s);
}

void r3d_stats_report_now(r3d_stats *s) {
  if (s->count == 0) return;
  uint64_t cpu_sum = 0, gpu_sum = 0, cpu_max = 0;
  for (uint32_t i = 0; i < s->count; i++) {
    cpu_sum += s->cpu_ns[i];
    gpu_sum += s->gpu_ns[i];
    if (s->cpu_ns[i] > cpu_max) cpu_max = s->cpu_ns[i];
  }
  double cpu_ms = (double)cpu_sum / (double)s->count / 1e6;
  double gpu_ms = (double)gpu_sum / (double)s->count / 1e6;
  double max_ms = (double)cpu_max / 1e6;
  double fps = cpu_ms > 0.0 ? 1000.0 / cpu_ms : 0.0;
  printf("frame %8llu | %6.1f fps | cpu %6.2f ms (max %6.2f) | gpu %6.2f ms\n",
         (unsigned long long)s->frame_index, fps, cpu_ms, max_ms, gpu_ms);
  fflush(stdout);
}
