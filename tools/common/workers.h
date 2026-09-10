/* Reusable phase workers: callers bound produced payloads per phase and
 * consume them after run() returns. No worker outlives destroy(). */
#ifndef R3D_TOOL_WORKERS_H
#define R3D_TOOL_WORKERS_H
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#define R3D_TOOL_MAX_WORKERS 32u
typedef struct r3d_tool_workers {
  pthread_t threads[R3D_TOOL_MAX_WORKERS];
  pthread_mutex_t mu;
  pthread_cond_t start, done;
  uint32_t count, pending;
  uint64_t generation;
  bool stop;
  void *(*work)(void *);
  void *arg;
  void (*cleanup)(void); /* optional thread-local cleanup, set before first run */
} r3d_tool_workers;
static void *r3d_tool_worker_main(void *arg) {
  r3d_tool_workers *p = arg;
  uint64_t seen = 0;
  pthread_mutex_lock(&p->mu);
  for (;;) {
    while (!p->stop && p->generation == seen) pthread_cond_wait(&p->start, &p->mu);
    if (p->stop) break;
    seen = p->generation;
    void *(*work)(void *) = p->work;
    void *job = p->arg;
    pthread_mutex_unlock(&p->mu);
    work(job);
    pthread_mutex_lock(&p->mu);
    if (--p->pending == 0) pthread_cond_signal(&p->done);
  }
  pthread_mutex_unlock(&p->mu);
  if (p->cleanup) p->cleanup();
  return NULL;
}
static void r3d_tool_workers_destroy(r3d_tool_workers *p) {
  pthread_mutex_lock(&p->mu);
  p->stop = true;
  pthread_cond_broadcast(&p->start);
  pthread_mutex_unlock(&p->mu);
  for (uint32_t i = 0; i < p->count; i++) pthread_join(p->threads[i], NULL);
  pthread_cond_destroy(&p->start);
  pthread_cond_destroy(&p->done);
  pthread_mutex_destroy(&p->mu);
}
static int r3d_tool_workers_init(r3d_tool_workers *p, uint32_t n) {
  memset(p, 0, sizeof *p);
  if (!n) n = 1;
  if (n > R3D_TOOL_MAX_WORKERS) n = R3D_TOOL_MAX_WORKERS;
  pthread_mutex_init(&p->mu, NULL);
  pthread_cond_init(&p->start, NULL);
  pthread_cond_init(&p->done, NULL);
  for (; p->count < n; p->count++)
    if (pthread_create(&p->threads[p->count], NULL, r3d_tool_worker_main, p) != 0) {
      r3d_tool_workers_destroy(p);
      return -1;
    }
  return 0;
}
static void r3d_tool_workers_run(r3d_tool_workers *p, void *(*work)(void *), void *arg) {
  pthread_mutex_lock(&p->mu);
  p->work = work;
  p->arg = arg;
  p->pending = p->count;
  p->generation++;
  pthread_cond_broadcast(&p->start);
  while (p->pending) pthread_cond_wait(&p->done, &p->mu);
  pthread_mutex_unlock(&p->mu);
}
#endif
