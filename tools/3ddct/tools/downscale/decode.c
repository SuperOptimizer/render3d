// decode.c — see decode.h.

#include "decode.h"

#include <string.h>

#include "../export/shard.h"
#include "../../dct3d.h"

static uint64_t get_u64_le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

// Scatter one decoded 16^3 block into `vol`, a dense S_Z x SHARD_VOX x
// SHARD_VOX buffer whose z=0 corresponds to source inner-chunk row `iz_base`
// (0 for a full-shard decode, iz0 for a Z-slab decode).
static void scatter_block(uint8_t *vol, int iz, int iy, int ix, int iz_base,
                          const uint8_t *blk) {
    const size_t S = SHARD_VOX;
    const int z0 = (iz - iz_base) * INNER, y0 = iy * INNER, x0 = ix * INNER;
    for (int z = 0; z < INNER; ++z)
        for (int y = 0; y < INNER; ++y) {
            uint8_t *dst = vol + (((size_t)(z0 + z) * S + (y0 + y)) * S + x0);
            memcpy(dst, blk + (size_t)(z * INNER + y) * INNER, INNER);
        }
}

// Shared implementation: decode inner-chunk Z-rows [iz0, iz1) of `blob` into
// `vol` (a ((iz1-iz0)*INNER) x SHARD_VOX x SHARD_VOX buffer, z=0 == source
// z = iz0*INNER). shard_decode_u8 is just this with the full [0, GRID) range.
static int decode_zrows_impl(const uint8_t *blob, size_t len, int iz0, int iz1,
                             uint8_t *vol) {
    const size_t idx_bytes = NINNER * 16 + 4;
    if (len < idx_bytes) return -1;
    const uint8_t *idx = blob + (len - idx_bytes);

    uint32_t want_crc = 0;
    for (int i = 3; i >= 0; --i) want_crc = (want_crc << 8) | idx[NINNER * 16 + i];
    if (shard_crc32c(idx, NINNER * 16) != want_crc) return -1;

    uint8_t blk[DCT3D_N3];
    for (int iz = iz0; iz < iz1; ++iz)
        for (int iy = 0; iy < GRID; ++iy)
            for (int ix = 0; ix < GRID; ++ix) {
                const size_t ci = ((size_t)iz * GRID + iy) * GRID + ix;
                uint64_t off = get_u64_le(idx + ci * 16 + 0);
                uint64_t n = get_u64_le(idx + ci * 16 + 8);
                if (off == UINT64_MAX && n == UINT64_MAX) continue;  // missing == air; leave vol as-is
                if (off + n > len - idx_bytes) return -1;  // corrupt offset
                if (!dct3d_decode_u8(blob + off, (size_t)n, blk)) return -1;
                scatter_block(vol, iz, iy, ix, iz0, blk);
            }
    return 0;
}

int shard_decode_u8(const uint8_t *blob, size_t len, uint8_t *vol) {
    return decode_zrows_impl(blob, len, 0, GRID, vol);
}

int shard_decode_u8_zrows(const uint8_t *blob, size_t len, int iz0, int iz1,
                          uint8_t *vol) {
    return decode_zrows_impl(blob, len, iz0, iz1, vol);
}
