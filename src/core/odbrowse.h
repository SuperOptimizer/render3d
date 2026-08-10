/* Anonymous S3 bucket browsing (AWS open-data). One blocking HTTPS call per
 * listing: ListObjectsV2 with delimiter=/ returns the "directories"
 * (CommonPrefixes) and files (Contents keys) directly under a prefix. */
#ifndef R3D_ODBROWSE_H
#define R3D_ODBROWSE_H

#include <stdint.h>

typedef struct r3d_odlist {
  char **dirs;  /* prefix names relative to the queried prefix, no trailing / */
  char **files; /* file names relative to the queried prefix */
  uint64_t *file_sizes;
  uint32_t ndirs, nfiles;
} r3d_odlist;

/* bucket_url e.g. "https://vesuvius-challenge-open-data.s3.amazonaws.com",
 * prefix e.g. "PHercParis4/volumes/" (empty string lists the root).
 * Returns 0 on success. */
int r3d_odlist_fetch(const char *bucket_url, const char *prefix, r3d_odlist *out);
void r3d_odlist_free(r3d_odlist *l);

#endif /* R3D_ODBROWSE_H */
