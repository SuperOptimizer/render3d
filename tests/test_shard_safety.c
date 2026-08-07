#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "shard/shardio.h"

static int failures;
#define CHECK(c)                                                                            \
  do {                                                                                      \
    if (!(c)) {                                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                        \
      failures++;                                                                           \
    }                                                                                       \
  } while (0)

static uint32_t crc32c(const uint8_t *p, size_t n) {
  uint32_t c = UINT32_MAX;
  while (n--) {
    c ^= *p++;
    for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0x82F63B78u & (uint32_t)-(int)(c & 1));
  }
  return c ^ UINT32_MAX;
}

static void put_u32le(uint8_t *p, uint32_t v) {
  for (unsigned i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8u * i));
}

static void put_u64le(uint8_t *p, uint64_t v) {
  for (unsigned i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8u * i));
}

static int write_all(int fd, const uint8_t *p, size_t n) {
  while (n) {
    ssize_t w = write(fd, p, n);
    if (w <= 0) return -1;
    p += (size_t)w;
    n -= (size_t)w;
  }
  return 0;
}

static int write_fixture(const char *path, uint8_t *index, int bad_crc) {
  uint32_t crc = crc32c(index, R3D_SHARD_NCHUNKS * 16u);
  put_u32le(index + R3D_SHARD_NCHUNKS * 16u, crc ^ (bad_crc ? 1u : 0u));
  int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
  if (fd < 0) return -1;
  int rc = write_all(fd, index, R3D_SHARD_INDEX_BYTES);
  if (close(fd) != 0) rc = -1;
  return rc;
}

int main(void) {
  char dir[] = "/tmp/render3d-shard-test-XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  char path[256];
  snprintf(path, sizeof path, "%s/0_0_0.shard", dir);
  uint8_t *index = malloc(R3D_SHARD_INDEX_BYTES);
  CHECK(index != NULL);
  if (!index) return EXIT_FAILURE;
  memset(index, 0xff, R3D_SHARD_NCHUNKS * 16u);

  r3d_shard_store store;
  CHECK(r3d_shard_store_init(&store, dir, 1024, 1024, 1024) == 0);
  CHECK(write_fixture(path, index, 0) == 0);
  r3d_shard sh;
  CHECK(r3d_shard_open(&store, 0, 0, 0, &sh) == R3D_SHARD_OK);
  uint8_t chunk[4096];
  memset(chunk, 0xa5, sizeof chunk);
  CHECK(r3d_shard_chunk_decode(&sh, 0, 0, 0, chunk) == 0);
  for (size_t i = 0; i < sizeof chunk; i++) CHECK(chunk[i] == 0);
  r3d_shard_close(&sh);

  CHECK(write_fixture(path, index, 1) == 0);
  CHECK(r3d_shard_open(&store, 0, 0, 0, &sh) == R3D_SHARD_CORRUPT);

  /* A valid index CRC must not legitimize an overflowing payload range. */
  put_u64le(index, UINT64_MAX - 7u);
  put_u64le(index + 8, 16);
  CHECK(write_fixture(path, index, 0) == 0);
  CHECK(r3d_shard_open(&store, 0, 0, 0, &sh) == R3D_SHARD_OK);
  CHECK(r3d_shard_chunk_decode(&sh, 0, 0, 0, chunk) == -1);
  r3d_shard_close(&sh);
  CHECK(r3d_shard_decode_region(&store, 0, 0, 0, 16, 16, 16, chunk, 2) == -1);

  CHECK(r3d_shard_decode_region(&store, 0, 0, 0, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                chunk, 1) == -1);

  free(index);
  unlink(path);
  rmdir(dir);
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  puts("test_shard_safety: all ok");
  return EXIT_SUCCESS;
}
