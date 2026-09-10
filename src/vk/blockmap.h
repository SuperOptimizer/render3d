/* Flat uint64 key / uint32 value map. UINT64_MAX is reserved for an empty key.
 * Deletion backshifts the cluster, so repeated slot churn never accumulates
 * tombstones. External synchronization is the caller's responsibility. */
#ifndef R3D_BLOCKMAP_H
#define R3D_BLOCKMAP_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define BR_MAP_EMPTY UINT32_MAX
#define BR_KEY_EMPTY UINT64_MAX
struct br_pair { uint64_t key; uint32_t value, pad; };
struct br_map { struct br_pair *entries; uint32_t capacity, count, cursor; };
static inline uint32_t br_hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
static inline uint32_t br_hash(uint64_t key) {
  return br_hash32((uint32_t)key ^ br_hash32((uint32_t)(key >> 32)));
}
static inline uint32_t br_map_find(const struct br_map *m, uint64_t key) {
  if (!m->capacity || key==BR_KEY_EMPTY) return BR_MAP_EMPTY;
  uint32_t mask=m->capacity-1, i=br_hash(key)&mask;
  for (uint32_t probes=0;probes<m->capacity;probes++,i=(i+1)&mask) {
    if (m->entries[i].key==key) return i;
    if (m->entries[i].key==BR_KEY_EMPTY) return BR_MAP_EMPTY;
  }
  return BR_MAP_EMPTY;
}
static inline uint32_t br_map_insert(struct br_map *m,uint64_t key,uint32_t value) {
  if (!m->capacity || key==BR_KEY_EMPTY) return BR_MAP_EMPTY;
  uint32_t mask=m->capacity-1,i=br_hash(key)&mask;
  for(uint32_t probes=0;probes<m->capacity;probes++,i=(i+1)&mask) {
    struct br_pair *e=&m->entries[i];
    if(e->key==key || e->key==BR_KEY_EMPTY) {
      if(e->key==BR_KEY_EMPTY)m->count++;
      e->key=key;e->value=value;return i;
    }
  }
  return BR_MAP_EMPTY;
}
/* visit reports every physical entry changed, for GPU dirty-word tracking. */
static inline void br_map_delete(struct br_map *m,uint64_t key,
                                void (*visit)(void *,uint32_t),void *user) {
  uint32_t hole=br_map_find(m,key);
  if(hole==BR_MAP_EMPTY)return;
  uint32_t mask=m->capacity-1;
  for(uint32_t j=(hole+1)&mask,steps=0;steps+1<m->capacity && m->entries[j].key!=BR_KEY_EMPTY;j=(j+1)&mask,steps++) {
    uint32_t home=br_hash(m->entries[j].key)&mask;
    if(((j-home)&mask)>=((j-hole)&mask)) {
      m->entries[hole]=m->entries[j];if(visit)visit(user,hole);hole=j;
    }
  }
  m->entries[hole]=(struct br_pair){BR_KEY_EMPTY,BR_MAP_EMPTY,0};
  if(visit)visit(user,hole);
  m->count--;
}
static inline int br_map_resize(struct br_map *m,uint32_t capacity) {
  if(capacity<4 || (capacity&(capacity-1)) || capacity<=m->count)return -1;
  struct br_pair *entries=malloc((size_t)capacity*sizeof *entries);
  if(!entries)return -1;
  memset(entries,0xff,(size_t)capacity*sizeof *entries);
  struct br_map next={.entries=entries,.capacity=capacity};
  for(uint32_t i=0;i<m->capacity;i++)if(m->entries[i].key!=BR_KEY_EMPTY)
    (void)br_map_insert(&next,m->entries[i].key,m->entries[i].value);
  free(m->entries);*m=next;return 0;
}
static inline void br_map_clear(struct br_map *m) {
  memset(m->entries,0xff,(size_t)m->capacity*sizeof *m->entries);m->count=0;
}
#endif
