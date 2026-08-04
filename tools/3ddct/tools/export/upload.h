// upload.h — SFTP a shard blob to the data server via libcurl.
//
// libs3 is S3-only, so uploads use libcurl directly with an sftp:// URL (curl on
// the target is built with SSH support). The remote directory tree is created
// as needed (CURLOPT_FTP_CREATE_MISSING_DIRS). Host-key checking uses the
// caller's known_hosts; credentials are user:pass.

#ifndef DCT3D_EXPORT_UPLOAD_H
#define DCT3D_EXPORT_UPLOAD_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *host;       // dl.ash2txt.org
    int port;               // 9238
    const char *user;
    const char *pass;
    const char *known_hosts;  // path to known_hosts (NULL -> ~/.ssh/known_hosts)
} sftp_target;

// Upload `len` bytes to sftp://host:port/<remote_path>, creating missing dirs.
// `remote_path` is absolute from the SFTP root (leading '/'). Returns 0 on
// success, -1 on failure (message to stderr). Thread-safe: uses its own easy
// handle per call.
int sftp_upload(const sftp_target *t, const char *remote_path,
                const uint8_t *data, size_t len);

// Return the remote file size (>= 0) if `remote_path` exists, or -1 if it does
// not exist / cannot be stat'd. Used for resume (skip already-uploaded shards).
// Thread-safe.
long sftp_size(const sftp_target *t, const char *remote_path);

// One-time global libcurl init (call once before threads start).
void sftp_global_init(void);

#endif  // DCT3D_EXPORT_UPLOAD_H
