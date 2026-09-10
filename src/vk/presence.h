/* Bounded source-availability cache. Eviction forgets only a hint: persistent
 * compressed files remain authoritative and are rediscovered by workers. */
#ifndef R3D_PRESENCE_H
#define R3D_PRESENCE_H
#include "blockmap.h"
#include <pthread.h>
struct ni_presence { pthread_mutex_t mu; struct br_map map; };
static inline struct ni_presence *ni_presence_new(void) {
  struct ni_presence *p=calloc(1,sizeof *p);
  if(p && pthread_mutex_init(&p->mu,NULL)){free(p);return NULL;}
  return p;
}
static inline void ni_presence_free(struct ni_presence *p) {
  if(!p)return;
  pthread_mutex_destroy(&p->mu);free(p->map.entries);free(p);
}
static inline void ni_presence_clear(struct ni_presence *p) {
  if(!p)return;
  pthread_mutex_lock(&p->mu);
  if(p->map.capacity)br_map_clear(&p->map);
  pthread_mutex_unlock(&p->mu);
}
static inline uint8_t ni_presence_get(struct ni_presence *p,uint64_t key) {
  if(!p)return 0;
  pthread_mutex_lock(&p->mu);
  uint32_t i=br_map_find(&p->map,key);
  uint8_t value=i==BR_MAP_EMPTY?0:(uint8_t)p->map.entries[i].value;
  pthread_mutex_unlock(&p->mu);return value;
}
static inline void ni_presence_set(struct ni_presence *p,uint64_t key,uint32_t value) {
  if(!p)return;
  pthread_mutex_lock(&p->mu);
  struct br_map *m=&p->map;
  if(!m->capacity && br_map_resize(m,1024u))goto done;
  if(br_map_find(m,key)==BR_MAP_EMPTY && m->count>=m->capacity/2u) {
    if(m->capacity>=(1u<<18) || br_map_resize(m,m->capacity*2u)) {
      for(uint32_t n=0;n<m->capacity;n++) {
        uint32_t i=m->cursor++&(m->capacity-1u);
        if(m->entries[i].key!=BR_KEY_EMPTY){br_map_delete(m,m->entries[i].key,NULL,NULL);break;}
      }
    }
  }
  (void)br_map_insert(m,key,value);
done:
  pthread_mutex_unlock(&p->mu);
}
#endif
