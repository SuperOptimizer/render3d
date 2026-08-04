// decode.h — unpack a Zarr v3 sharding_indexed dct3d shard back to a dense
// SHARD_VOX^3 u8 volume (or a Z-slab of it). Inverse of shard.c's
// shard_encode_u8.

#ifndef DCT3D_DOWNSCALE_DECODE_H
#define DCT3D_DOWNSCALE_DECODE_H

#include <stddef.h>
#include <stdint.h>

// Decode `blob` (len bytes, as produced by shard_encode_u8 / read verbatim off
// disk) into `vol` (SHARD_VOX^3, caller-allocated, z-major). Missing/sentinel
// inner chunks are left as their pre-existing `vol` contents (caller should
// zero `vol` first for "missing == air" semantics). Returns 0 on success, -1 on
// a corrupt/truncated blob (bad crc or index).
int shard_decode_u8(const uint8_t *blob, size_t len, uint8_t *vol);

// Decode only the inner-chunk Z-rows [iz0, iz1) (0 <= iz0 < iz1 <= GRID) of
// `blob` into `vol`, a dense ((iz1-iz0)*INNER) x SHARD_VOX x SHARD_VOX u8
// buffer, z-major, z=0 of `vol` corresponding to source z = iz0*INNER. This is
// the streaming counterpart to shard_decode_u8: it still parses/validates the
// full index+crc trailer (cheap — NINNER*16 bytes) but only runs
// dct3d_decode_u8 on the ~(iz1-iz0)*GRID*GRID inner chunks that fall in the
// requested Z range, so peak memory for a partial decode is proportional to
// the slab thickness, not the whole SHARD_VOX^3 volume. `vol` should be
// zeroed by the caller first (same missing==air convention as
// shard_decode_u8). Returns 0 on success, -1 on a corrupt/truncated blob.
int shard_decode_u8_zrows(const uint8_t *blob, size_t len, int iz0, int iz1,
                          uint8_t *vol);

#endif  // DCT3D_DOWNSCALE_DECODE_H
