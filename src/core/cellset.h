/* Insertion-ordered XYZ source-cell set. Coordinates are not bit-packed:
 * every uint32 coordinate is preserved, including values above 21 bits. */
#ifndef R3D_CELLSET_H
#define R3D_CELLSET_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct r3d_cellset {
  uint32_t (*cells)[3], *slots;
  uint32_t n, capacity, nslots;
} r3d_cellset;
static inline uint32_t r3d_cell_hash(uint32_t x, uint32_t y, uint32_t z) {
  uint64_t h = (uint64_t)x * 0x9e3779b185ebca87ull ^
               (uint64_t)y * 0xc2b2ae3d27d4eb4full ^
               (uint64_t)z * 0x165667b19e3779f9ull;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdull;
  h ^= h >> 33;
  return (uint32_t)h;
}
static inline void r3d_cellset_free(r3d_cellset *s) {
  free(s->cells);
  free(s->slots);
  memset(s, 0, sizeof *s);
}
static inline void r3d_cellset_clear(r3d_cellset *s) {
  s->n = 0;
  if (s->slots)
    memset(s->slots, 0, (size_t)s->nslots * sizeof *s->slots);
}
/* 1=new, 0=already present, -1=allocation failure, -2=unique-cell budget.
 * A caller can flush and clear on -2: no source cell is silently dropped. */
static inline int r3d_cellset_add(r3d_cellset *s, uint32_t x, uint32_t y,
                                  uint32_t z, uint32_t limit) {
  uint32_t h = r3d_cell_hash(x, y, z), slot = 0;
  if (s->nslots) {
    slot = h & (s->nslots - 1);
    while (s->slots[slot]) {
      const uint32_t *c = s->cells[s->slots[slot] - 1];
      if (c[0] == x && c[1] == y && c[2] == z)
        return 0;
      slot = (slot + 1) & (s->nslots - 1);
    }
  }
  if (s->n >= limit)
    return -2;
  if (s->n == s->capacity) {
    uint32_t cap = s->capacity ? s->capacity * 2 : 256;
    if (cap < s->capacity || cap > UINT32_MAX / 2)
      return -1;
    uint32_t (*cells)[3] = realloc(s->cells, (size_t)cap * sizeof *cells);
    if (!cells)
      return -1;
    s->cells = cells;
    s->capacity = cap;
  }
  if (s->nslots == 0 || s->n >= s->nslots / 2) {
    uint32_t count = s->nslots ? s->nslots * 2 : 512;
    if (count < s->nslots)
      return -1;
    uint32_t *slots = calloc(count, sizeof *slots);
    if (!slots)
      return -1;
    for (uint32_t i = 0; i < s->n; i++) {
      uint32_t j =
          r3d_cell_hash(s->cells[i][0], s->cells[i][1], s->cells[i][2]) &
          (count - 1);
      while (slots[j])
        j = (j + 1) & (count - 1);
      slots[j] = i + 1;
    }
    free(s->slots);
    s->slots = slots;
    s->nslots = count;
    slot = h & (count - 1);
    while (slots[slot])
      slot = (slot + 1) & (count - 1);
  }
  s->cells[s->n][0] = x;
  s->cells[s->n][1] = y;
  s->cells[s->n][2] = z;
  s->slots[slot] = ++s->n;
  return 1;
}
#endif
