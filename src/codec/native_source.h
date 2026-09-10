#ifndef R3D_NATIVE_SOURCE_H
#define R3D_NATIVE_SOURCE_H
#include <curl/curl.h>
#include <stdint.h>
#include <stdbool.h>
/* Fetch one native 128^3 stream from a Zarr-v3 1024^3 indexed shard.
 * Coordinates are 128^3 chunk indices. Publishes atomically; returns 1 for
 * payload, 2 for known air, 0 for failure. Reuses curl's connection cache.
 * Optional downloaded distinguishes a successful transfer from a cache hit. */
int r3d_native_fetch(CURL *curl, const char *url, const char *root,
                     uint32_t level, uint32_t x, uint32_t y, uint32_t z, bool *downloaded);
#endif
