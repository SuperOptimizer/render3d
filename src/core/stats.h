/* Frame-time ring buffer + 1 Hz stdout report. Times in nanoseconds. */
#ifndef R3D_STATS_H
#define R3D_STATS_H

#include <stdint.h>

#define R3D_STATS_RING 240

typedef struct r3d_stats {
  uint64_t cpu_ns[R3D_STATS_RING]; /* whole-frame CPU time   */
  uint64_t gpu_ns[R3D_STATS_RING]; /* raycast dispatch time  */
  uint32_t head, count;
  uint64_t last_report_ns; /* monotonic time of last stdout report */
  uint64_t frame_index;
} r3d_stats;

uint64_t r3d_now_ns(void); /* CLOCK_MONOTONIC */

void r3d_stats_init(r3d_stats *s);
void r3d_stats_push(r3d_stats *s, uint64_t cpu_ns, uint64_t gpu_ns);
/* Prints one line at most once per second: fps, mean/p99 cpu ms, mean gpu ms. */
void r3d_stats_report(r3d_stats *s);

#endif /* R3D_STATS_H */
