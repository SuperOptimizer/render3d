#include "core/stats.h"

#include <stdio.h>
#include <stdlib.h>
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

static int u64_cmp(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return x < y ? -1 : x > y;
}

void r3d_stats_summarize_values(const uint64_t *values, uint32_t count,
                                r3d_stats_summary *out) {
  memset(out, 0, sizeof *out);
  if (!values || count == 0) return;
  uint64_t *sorted = malloc((size_t)count * sizeof *sorted);
  if (!sorted) return;
  memcpy(sorted, values, (size_t)count * sizeof *sorted);
  qsort(sorted, count, sizeof *sorted, u64_cmp);
  long double sum = 0;
  for (uint32_t i = 0; i < count; i++) sum += sorted[i];
  out->mean_ns = (double)(sum / count);
  out->p50_ns = sorted[((uint64_t)count * 50u + 99u) / 100u - 1u];
  out->p95_ns = sorted[((uint64_t)count * 95u + 99u) / 100u - 1u];
  out->p99_ns = sorted[((uint64_t)count * 99u + 99u) / 100u - 1u];
  out->max_ns = sorted[count - 1u];
  free(sorted);
}

void r3d_stats_summarize(const r3d_stats *s, r3d_stats_summary *cpu,
                         r3d_stats_summary *gpu) {
  r3d_stats_summarize_values(s->cpu_ns, s->count, cpu);
  r3d_stats_summarize_values(s->gpu_ns, s->count, gpu);
}

void r3d_stats_report_now(r3d_stats *s) {
  if (s->count == 0) return;
  r3d_stats_summary cpu, gpu;
  r3d_stats_summarize(s, &cpu, &gpu);
  double cpu_ms = cpu.mean_ns / 1e6;
  double fps = cpu_ms > 0.0 ? 1000.0 / cpu_ms : 0.0;
  printf("frame %8llu | %6.1f fps | cpu %.2f ms p95 %.2f p99 %.2f max %.2f | "
         "gpu %.2f ms p95 %.2f p99 %.2f max %.2f\n",
         (unsigned long long)s->frame_index, fps, cpu_ms, (double)cpu.p95_ns / 1e6,
         (double)cpu.p99_ns / 1e6, (double)cpu.max_ns / 1e6, gpu.mean_ns / 1e6,
         (double)gpu.p95_ns / 1e6, (double)gpu.p99_ns / 1e6, (double)gpu.max_ns / 1e6);
  fflush(stdout);
}
