// shard.c — see shard.h. Zarr v3 sharding_indexed packing of dct3d chunks.

#include "shard.h"

#include <stdlib.h>
#include <string.h>

// crc32c (Castagnoli), bit-reflected, poly 0x82F63B78, init/xorout 0xFFFFFFFF.
// Matches zarr's default index crc32c trailer (verified byte-for-byte against
// zarr-python output). Table built on first use.
static uint32_t g_crc_tab[256];
static int g_crc_ready = 0;

static void crc_init(void) {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0x82F63B78u & (uint32_t)(-(int)(c & 1)));
        g_crc_tab[i] = c;
    }
    g_crc_ready = 1;
}

uint32_t shard_crc32c(const uint8_t *data, size_t len) {
    if (!g_crc_ready) crc_init();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) crc = g_crc_tab[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

static void put_u64_le(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static void put_u32_le(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
}

// Gather one 16^3 inner chunk (iz,iy,ix) out of the dense SHARD_VOX^3 volume.
static void gather_block(const uint8_t *vol, int iz, int iy, int ix, uint8_t *blk) {
    const size_t S = SHARD_VOX;
    const int z0 = iz * INNER, y0 = iy * INNER, x0 = ix * INNER;
    for (int z = 0; z < INNER; ++z)
        for (int y = 0; y < INNER; ++y) {
            const uint8_t *src = vol + (((size_t)(z0 + z) * S + (y0 + y)) * S + x0);
            memcpy(blk + (size_t)(z * INNER + y) * INNER, src, INNER);
        }
}

static int block_all_zero(const uint8_t *blk) {
    for (size_t i = 0; i < (size_t)INNER * INNER * INNER; ++i)
        if (blk[i]) return 0;
    return 1;
}

int shard_encode_u8(const uint8_t *vol, float quality, float tau,
                    uint8_t **out, size_t *out_len) {
    // Worst case per chunk = DCT3D_MAX_BYTES; index = NINNER*16 + 4 crc.
    const size_t idx_bytes = NINNER * 16 + 4;
    // Payload upper bound. dct3d rarely approaches DCT3D_MAX_BYTES, but size for
    // the worst case so we never realloc mid-encode.
    size_t cap = NINNER * (size_t)DCT3D_MAX_BYTES + idx_bytes;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return -1;

    uint8_t blk[DCT3D_N3];
    uint8_t enc[DCT3D_MAX_BYTES];
    uint8_t *idx = (uint8_t *)malloc(NINNER * 16);
    if (!idx) { free(buf); return -1; }

    size_t off = 0;  // running write offset into buf (== payload cursor)
    for (int iz = 0; iz < GRID; ++iz)
        for (int iy = 0; iy < GRID; ++iy)
            for (int ix = 0; ix < GRID; ++ix) {
                const size_t ci = ((size_t)iz * GRID + iy) * GRID + ix;
                gather_block(vol, iz, iy, ix, blk);
                if (block_all_zero(blk)) {
                    // Missing sentinel: (0xff..ff, 0xff..ff) -> fill_value on read.
                    memset(idx + ci * 16, 0xff, 16);
                    continue;
                }
                size_t n = dct3d_encode_u8(blk, quality, 0.0f, tau, enc);
                memcpy(buf + off, enc, n);
                put_u64_le(idx + ci * 16 + 0, (uint64_t)off);
                put_u64_le(idx + ci * 16 + 8, (uint64_t)n);
                off += n;
            }

    // Append index + crc32c(index).
    memcpy(buf + off, idx, NINNER * 16);
    uint32_t crc = shard_crc32c(idx, NINNER * 16);
    put_u32_le(buf + off + NINNER * 16, crc);
    off += NINNER * 16 + 4;

    free(idx);
    *out = buf;
    *out_len = off;
    return 0;
}
