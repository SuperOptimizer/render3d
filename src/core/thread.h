#ifndef R3D_THREAD_H
#define R3D_THREAD_H
#include <pthread.h>
/* Numeric workers use the same 8 MiB virtual stack budget on every platform.
 * macOS defaults to 512 KiB, insufficient for the tracer's numeric scratch. */
static inline int r3d_thread_create(pthread_t *thread,
                                    const pthread_attr_t *attr,
                                    void *(*fn)(void *), void *arg) {
  if (attr)
    return pthread_create(thread, attr, fn, arg);
  pthread_attr_t a;
  int rc = pthread_attr_init(&a);
  if (rc)
    return rc;
  rc = pthread_attr_setstacksize(&a, 8u << 20);
  if (!rc)
    rc = pthread_create(thread, &a, fn, arg);
  pthread_attr_destroy(&a);
  return rc;
}
#endif
