// localio.h — local-filesystem analogs of the SFTP get/put/size helpers, for
// the sneakernet workflow: shards are produced on a local NVMe (--local-root)
// and physically carried to fast internet for upload later. Layout under the
// root mirrors the server's /exports tree exactly so the eventual upload is a
// plain recursive sync.

#ifndef DCT3D_DOWNSCALE_LOCALIO_H
#define DCT3D_DOWNSCALE_LOCALIO_H

#include <stddef.h>
#include <stdint.h>

// Size in bytes of a local file, or -1 if it does not exist.
long local_size(const char *path);

// Read an entire local file into a malloc'd buffer. 0 on success.
int local_read(const char *path, uint8_t **out, size_t *out_len);

// Write buf to path atomically (temp file in the same dir + rename), creating
// parent directories as needed. 0 on success.
int local_write_atomic(const char *path, const uint8_t *buf, size_t len);

#endif  // DCT3D_DOWNSCALE_LOCALIO_H
