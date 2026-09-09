/* Exact painted label IDs, stored with zlib; probability volumes use volcomp.
 */
#ifndef R3D_CODEC_LABEL_H
#define R3D_CODEC_LABEL_H
#include <stddef.h>
#include <stdint.h>
typedef enum r3d_label_type { R3D_LABEL_U8 = 0 } r3d_label_type;
#define R3D_LABEL_NO_MASK UINT32_MAX
#define R3D_LABEL_MAX_CHANNELS 1u
typedef struct r3d_label_channel {
  r3d_label_type type;
  uint32_t mask_chan;
  void *data;
} r3d_label_channel;
typedef struct r3d_label_params {
  unsigned nthreads;
} r3d_label_params;
r3d_label_params r3d_label_defaults(void);
int r3d_label_encode(const r3d_label_params *p, const r3d_label_channel *ch,
                     uint32_t nc, uint32_t dim, uint8_t **out, size_t *n);
int r3d_label_info(const uint8_t *in, size_t n, uint32_t *dim, uint32_t *nc,
                   r3d_label_type *types, uint32_t *masks);
int r3d_label_decode(const uint8_t *in, size_t n, uint32_t dim,
                     r3d_label_channel *ch, uint32_t nc, unsigned threads);
#endif
