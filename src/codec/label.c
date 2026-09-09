#include "label.h"
#include "bytes.h"
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
r3d_label_params r3d_label_defaults(void) { return (r3d_label_params){1}; }
int r3d_label_encode(const r3d_label_params *p, const r3d_label_channel *ch,
                     uint32_t nc, uint32_t dim, uint8_t **out, size_t *n) {
  (void)p;
  if (!out || !n)
    return -1;
  *out = NULL;
  *n = 0;
  if (!ch || nc != 1 || dim != 128 || ch->type != R3D_LABEL_U8 ||
      ch->mask_chan != R3D_LABEL_NO_MASK || !ch->data)
    return -1;
  uLongf cap = compressBound(128u * 128u * 128u);
  uint8_t *buf = malloc((size_t)cap + 8);
  if (!buf)
    return -1;
  memcpy(buf, "R3L1", 4);
  r3c_put32(buf + 4, dim);
  if (compress2(buf + 8, &cap, ch->data, 128u * 128u * 128u, Z_BEST_SPEED) !=
      Z_OK) {
    free(buf);
    return -1;
  }
  *out = buf;
  *n = (size_t)cap + 8;
  return 0;
}
int r3d_label_info(const uint8_t *in, size_t n, uint32_t *dim, uint32_t *nc,
                   r3d_label_type *types, uint32_t *masks) {
  if (!in || n <= 8 || memcmp(in, "R3L1", 4) || r3c_u32(in + 4) != 128 ||
      !dim || !nc)
    return -1;
  *dim = 128;
  *nc = 1;
  if (types)
    types[0] = R3D_LABEL_U8;
  if (masks)
    masks[0] = R3D_LABEL_NO_MASK;
  return 0;
}
int r3d_label_decode(const uint8_t *in, size_t n, uint32_t dim,
                     r3d_label_channel *ch, uint32_t nc, unsigned threads) {
  (void)threads;
  uint32_t dd, cc;
  if (r3d_label_info(in, n, &dd, &cc, NULL, NULL) || dd != dim || cc != nc ||
      !ch || ch->type != R3D_LABEL_U8 || ch->mask_chan != R3D_LABEL_NO_MASK ||
      !ch->data)
    return -1;
  uLongf len = 128u * 128u * 128u;
  uLong src = (uLong)(n - 8);
  return uncompress2(ch->data, &len, in + 8, &src) == Z_OK &&
                 len == 128u * 128u * 128u && src == n - 8
             ? 0
             : -1;
}
