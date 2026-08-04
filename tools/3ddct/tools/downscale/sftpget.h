// sftpget.h — SFTP GET a remote file into memory via libcurl.
//
// Counterpart to upload.h's sftp_upload/sftp_size: this tool reads its own
// level-(N-1) shards back from dl.ash2txt.org to build level N.

#ifndef DCT3D_DOWNSCALE_SFTPGET_H
#define DCT3D_DOWNSCALE_SFTPGET_H

#include <stddef.h>
#include <stdint.h>

#include "../export/upload.h"  // sftp_target, sftp_global_init

// Fetch remote_path into a malloc'd buffer (*out, caller frees), length *out_len.
// Returns 0 on success, -1 on failure (including 404 / does-not-exist), in which
// case *out is NULL. Thread-safe: each thread reuses ONE persistent connection
// across calls (the SSH handshake is paid once per thread, not per file).
int sftp_download(const sftp_target *t, const char *remote_path,
                  uint8_t **out, size_t *out_len);

// Size of a remote file, or -1 if absent/error. Same per-thread persistent
// connection as sftp_download (unlike upload.h's sftp_size, which handshakes
// per call).
long sftp_stat_size(const sftp_target *t, const char *remote_path);

#endif  // DCT3D_DOWNSCALE_SFTPGET_H
