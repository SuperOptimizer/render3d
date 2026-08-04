// upload.c — see upload.h.

#include "upload.h"

#include <curl/curl.h>
#include <stdio.h>
#include <string.h>

void sftp_global_init(void) { curl_global_init(CURL_GLOBAL_DEFAULT); }

typedef struct { const uint8_t *data; size_t len, pos; } read_ctx;

static size_t read_cb(char *buffer, size_t size, size_t nitems, void *userdata) {
    read_ctx *rc = (read_ctx *)userdata;
    size_t want = size * nitems;
    size_t left = rc->len - rc->pos;
    size_t n = want < left ? want : left;
    if (n) { memcpy(buffer, rc->data + rc->pos, n); rc->pos += n; }
    return n;
}

int sftp_upload(const sftp_target *t, const char *remote_path,
                const uint8_t *data, size_t len) {
    CURL *h = curl_easy_init();
    if (!h) { fprintf(stderr, "sftp: curl_easy_init failed\n"); return -1; }

    char url[2048];
    // remote_path starts with '/', absolute from the server root.
    snprintf(url, sizeof(url), "sftp://%s:%d%s", t->host, t->port, remote_path);

    char userpwd[512];
    snprintf(userpwd, sizeof(userpwd), "%s:%s", t->user, t->pass);

    read_ctx rc = {data, len, 0};

    curl_easy_setopt(h, CURLOPT_URL, url);
    curl_easy_setopt(h, CURLOPT_USERPWD, userpwd);
    curl_easy_setopt(h, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(h, CURLOPT_READFUNCTION, read_cb);
    curl_easy_setopt(h, CURLOPT_READDATA, &rc);
    curl_easy_setopt(h, CURLOPT_INFILESIZE_LARGE, (curl_off_t)len);
    curl_easy_setopt(h, CURLOPT_FTP_CREATE_MISSING_DIRS, CURLFTP_CREATE_DIR_RETRY);
    if (t->known_hosts)
        curl_easy_setopt(h, CURLOPT_SSH_KNOWNHOSTS, t->known_hosts);
    curl_easy_setopt(h, CURLOPT_SSH_AUTH_TYPES, CURLSSH_AUTH_PASSWORD);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1L);   // stall watchdog
    curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, 120L);

    CURLcode res = CURLE_OK;
    for (int attempt = 0; attempt < 4; ++attempt) {
        rc.pos = 0;
        res = curl_easy_perform(h);
        if (res == CURLE_OK) break;
        fprintf(stderr, "sftp: upload %s attempt %d failed: %s\n",
                remote_path, attempt + 1, curl_easy_strerror(res));
    }
    curl_easy_cleanup(h);
    return res == CURLE_OK ? 0 : -1;
}

long sftp_size(const sftp_target *t, const char *remote_path) {
    CURL *h = curl_easy_init();
    if (!h) return -1;

    char url[2048], userpwd[512];
    snprintf(url, sizeof(url), "sftp://%s:%d%s", t->host, t->port, remote_path);
    snprintf(userpwd, sizeof(userpwd), "%s:%s", t->user, t->pass);

    // NOBODY + FILETIME triggers an SFTP stat without transferring the file.
    curl_easy_setopt(h, CURLOPT_URL, url);
    curl_easy_setopt(h, CURLOPT_USERPWD, userpwd);
    curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(h, CURLOPT_FILETIME, 1L);
    if (t->known_hosts)
        curl_easy_setopt(h, CURLOPT_SSH_KNOWNHOSTS, t->known_hosts);
    curl_easy_setopt(h, CURLOPT_SSH_AUTH_TYPES, CURLSSH_AUTH_PASSWORD);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 30L);

    long size = -1;
    CURLcode res = curl_easy_perform(h);
    if (res == CURLE_OK) {
        curl_off_t len = -1;
        if (curl_easy_getinfo(h, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &len) == CURLE_OK &&
            len >= 0)
            size = (long)len;
        else
            size = 0;  // exists but size unknown
    }
    curl_easy_cleanup(h);
    return size;
}
