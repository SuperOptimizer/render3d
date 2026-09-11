/* MIT; 64x64 independently decodable image/surface blocks. */
#ifndef SURFCOMP_H
#define SURFCOMP_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define SFC_VERSION_MAJOR 1
#define SFC_VERSION_MINOR 0
#define SFC_VERSION_PATCH 0
#define SFC_VERSION_STRING "1.0.0"
#define SFC_FORMAT_VERSION 4u
/* Runtime library identity, independent of the container version. */
const char *sfc_version(void);
uint32_t sfc_format_version(void);
#define SFC_EDGE 64u
#define SFC_SAMPLES 4096u
#define SFC_BLOCK_BOUND 40000u
#define SFC_XYZ_BOUND 200000u
#define SFC_XYZ 4u
#define SFC_EXACT 1u
#define SFC_COORDINATE 2u
#define SFC_MAX_CHANNELS 4096u
typedef enum { SFC_U8 = 1, SFC_U16 = 2, SFC_F32 = 3 } sfc_dtype;
typedef enum {
  SFC_OK = 0,
  SFC_INVALID = -1,
  SFC_IO = -2,
  SFC_NOMEM = -3,
  SFC_LIMIT = -4
} sfc_status;
typedef struct {
  char name[64];
  uint64_t width, height;
  sfc_dtype dtype;
  uint32_t flags;
  uint32_t components,
      component;    /* 0/1 = scalar; otherwise contiguous image planes */
  double tolerance; /* per-component error, or Euclidean for SFC_XYZ; zero for
                       exact */
} sfc_channel;
/* Byte strides; source/destination describe a full 64x64 padded block.
 * A NULL validity mask means all valid. Invalid float coordinates decode -1.
 * These standalone packets are self-contained B641 modes 0/1/2. Shared-table
 * mode 3 is container-only; use sfc_read_block for container payloads.
 * Returned buffers are owned by the caller and freed using free(). */
int sfc_encode_block(const sfc_channel *, const void *, size_t pixel_stride,
                     size_t row_stride, const uint8_t *valid, uint8_t **encoded,
                     size_t *size);
int sfc_decode_block(const uint8_t *, size_t, sfc_dtype, void *,
                     size_t pixel_stride, size_t row_stride, uint8_t *valid);
/* Interleaved float32 XYZ, Euclidean tolerance in voxel units. Valid samples
 * must be finite with positive Z. One packet contains one 64x64 patch. */
int sfc_encode_xyz_block(const void *, size_t, size_t, const uint8_t *, double,
                         uint8_t **, size_t *);
int sfc_decode_xyz_block(const uint8_t *, size_t, void *, size_t, size_t,
                         uint8_t *);
size_t sfc_sample_size(sfc_dtype);
uint32_t sfc_crc32(const void *, size_t);
/* All operations are synchronous. No worker threads or internal task dispatch.
 * Stateless block calls and separate writer instances can be used concurrently.
 * Calls on a single writer must be serialized in the required block order.
 * read_at returns 0 only after reading exactly size bytes. Thread safety is
 * the callback's responsibility. Concurrent reads on one reader are supported
 * if its callback supports them. Close must not race with any read.
 * Backing bytes must remain unchanged until close.
 * The library never allocates a full index. */
typedef int (*sfc_read_at)(void *user, uint64_t offset, void *dst, size_t size);
typedef struct sfc_reader sfc_reader;
typedef struct sfc_writer sfc_writer;
int sfc_open(sfc_read_at, void *, uint64_t file_size, sfc_reader **);
int sfc_open_file(const char *, sfc_reader **);
void sfc_close(sfc_reader *);
uint32_t sfc_channel_count(const sfc_reader *);
const sfc_channel *sfc_channel_info(const sfc_reader *, uint32_t);
int sfc_find_channel(const sfc_reader *, const char *);
uint64_t sfc_metadata_size(const sfc_reader *);
int sfc_read_metadata(sfc_reader *, void *, size_t);
int sfc_read_block(sfc_reader *, uint32_t channel, uint64_t bx, uint64_t by,
                   void *, size_t pixel_stride, size_t row_stride,
                   uint8_t *valid);
/* first_channel identifies X followed by Y/Z; also reads legacy scalar XYZ. */
int sfc_read_xyz(sfc_reader *, uint32_t first_channel, uint64_t bx, uint64_t by,
                 void *, size_t pixel_stride, size_t row_stride,
                 uint8_t *valid);
int sfc_block_range(sfc_reader *, uint32_t channel, uint64_t bx, uint64_t by,
                    float *min, float *max);
/* Transactional ROI output. Scratch is bounded by the explicitly requested
 * ROI, never the full source; invalid/out-of-bounds requests fail. */
int sfc_read_region(sfc_reader *, uint32_t, uint64_t x, uint64_t y, uint32_t w,
                    uint32_t h, void *, size_t pixel_stride, size_t row_stride,
                    uint8_t *valid);
/* Optional lazy decode cache. The reader is borrowed and must outlive the cache.
 * byte_budget caps tile allocations (samples, masks and entry bookkeeping).
 * Fixed cache control storage and the reader's bounded decode scratch are extra.
 * No allocation depends on surface dimensions. A budget too small for one
 * requested tile returns SFC_LIMIT on read; zero is permitted.
 * Calls on one cache are safe concurrently and serialize each tile lookup,
 * decode and copy using C11 atomics. No threads are created. For parallel
 * decoding, callers may use separate caches sharing a reader. A read_at
 * callback must not reenter the same cache. Destroy must not race with calls.
 * No borrowed tile pointers escape: results belong to the caller.
 * Regions are in bounds, unpadded; masks are tightly packed w*h bytes.
 * Unlike sfc_read_region, cache reads stream into the destination: an I/O or
 * decode failure can leave earlier tiles written. Invalid arguments leave
 * output unchanged. Joint XYZ scalar reads share a single cached XYZ tile. */
typedef struct sfc_cache sfc_cache;
typedef struct {
  size_t byte_budget, resident_bytes, resident_tiles;
  uint64_t hits, misses, evictions;
} sfc_cache_stats;
int sfc_cache_create(sfc_reader *, size_t byte_budget, sfc_cache **);
void sfc_cache_destroy(sfc_cache *);
void sfc_cache_clear(sfc_cache *); /* frees tiles; preserves cumulative counters */
int sfc_cache_get_stats(sfc_cache *, sfc_cache_stats *);
int sfc_cache_read_region(sfc_cache *, uint32_t channel, uint64_t x, uint64_t y,
                          uint32_t w, uint32_t h, void *, size_t pixel_stride,
                          size_t row_stride, uint8_t *valid);
/* Reads contiguous F32 coordinate channels, including legacy scalar triples. */
int sfc_cache_read_xyz_region(sfc_cache *, uint32_t first_channel,
                              uint64_t x, uint64_t y, uint32_t w, uint32_t h,
                              void *, size_t pixel_stride, size_t row_stride,
                              uint8_t *valid);
/* Creates a new file exclusively; existing paths are never overwritten.
 * Writes container version 4 (readers also accept versions 1 through 3). Up to
 * 16 blocks are buffered for shared entropy tables; errors can occur when
 * flushing. Append blocks once in channel, block-row, block-column order.
 * Header/index storage is streamed. finish verifies completeness; cancel
 * removes output. */
int sfc_create(const char *, const sfc_channel *, uint32_t,
               const void *metadata, size_t, sfc_writer **);
int sfc_write_block(sfc_writer *, const void *, size_t pixel_stride,
                    size_t row_stride, const uint8_t *);
/* For a contiguous SFC_XYZ|SFC_COORDINATE descriptor triple (F32, components=3,
 * components numbered 0..2, equal Euclidean tolerances). Appends all three
 * channels for the current patch; caller iterates patches once, then aux
 * fields. */
int sfc_write_xyz_block(sfc_writer *, const void *, size_t, size_t,
                        const uint8_t *);
/* Always consumes the writer, on success AND failure. Do not call cancel
 * or any other writer API afterward. Before finish, cancel also consumes it. */
int sfc_finish(sfc_writer *);
void sfc_cancel(sfc_writer *);
#ifdef __cplusplus
}
#endif
#endif
