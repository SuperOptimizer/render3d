// bqueue.h — a bounded blocking queue of void* items for the export pipeline.
//
// Producers `push` (blocking when full → backpressure); consumers `pop`
// (blocking when empty). `close` signals no more items; once closed and drained,
// `pop` returns 0 so consumer threads exit. Single fixed-capacity ring buffer,
// one mutex + two condvars. Capacity bounds in-flight memory (each item is a
// ~1 GiB buffer), which is the whole point.

#ifndef DCT3D_EXPORT_BQUEUE_H
#define DCT3D_EXPORT_BQUEUE_H

#include <pthread.h>
#include <stdlib.h>

typedef struct {
    void **buf;
    size_t cap, head, count;
    int closed;
    pthread_mutex_t lock;
    pthread_cond_t not_full, not_empty;
} bqueue;

static inline void bq_init(bqueue *q, size_t cap) {
    q->buf = (void **)calloc(cap, sizeof(void *));
    q->cap = cap;
    q->head = q->count = 0;
    q->closed = 0;
    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->not_full, NULL);
    pthread_cond_init(&q->not_empty, NULL);
}

static inline void bq_destroy(bqueue *q) {
    free(q->buf);
    pthread_mutex_destroy(&q->lock);
    pthread_cond_destroy(&q->not_full);
    pthread_cond_destroy(&q->not_empty);
}

// Push an item, blocking while the queue is full. Returns 1 on success, 0 if the
// queue was closed (item not enqueued — caller must free it).
static inline int bq_push(bqueue *q, void *item) {
    pthread_mutex_lock(&q->lock);
    while (q->count == q->cap && !q->closed)
        pthread_cond_wait(&q->not_full, &q->lock);
    if (q->closed) { pthread_mutex_unlock(&q->lock); return 0; }
    q->buf[(q->head + q->count) % q->cap] = item;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
    return 1;
}

// Pop an item into *out, blocking while empty. Returns 1 on success, 0 when the
// queue is closed AND drained (consumers should then exit).
static inline int bq_pop(bqueue *q, void **out) {
    pthread_mutex_lock(&q->lock);
    while (q->count == 0 && !q->closed)
        pthread_cond_wait(&q->not_empty, &q->lock);
    if (q->count == 0) { pthread_mutex_unlock(&q->lock); return 0; }
    *out = q->buf[q->head];
    q->head = (q->head + 1) % q->cap;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->lock);
    return 1;
}

// Signal that no more items will be pushed; wake all waiters.
static inline void bq_close(bqueue *q) {
    pthread_mutex_lock(&q->lock);
    q->closed = 1;
    pthread_cond_broadcast(&q->not_full);
    pthread_cond_broadcast(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
}

#endif  // DCT3D_EXPORT_BQUEUE_H
