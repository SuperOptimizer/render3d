// sftpget.c — see sftpget.h.
//
// Each thread keeps ONE persistent CURL easy handle (thread-local) and reuses
// it across requests: libcurl then reuses the underlying SSH session for
// subsequent sftp:// URLs on the same host+credentials, which matters a lot —
// a fresh SSH handshake costs ~1s per file, comparable to the transfer time
// of a typical parent blob. On a failed transfer the handle is torn down and
// rebuilt (the connection may be dead).

#include "sftpget.h"

#include <curl/curl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint8_t *buf; size_t len, cap; } grow_buf;

static size_t write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
    grow_buf *g = (grow_buf *)userdata;
    size_t n = size * nmemb;
    if (g->len + n > g->cap) {
        size_t ncap = g->cap ? g->cap * 2 : 1 << 20;
        while (ncap < g->len + n) ncap *= 2;
        uint8_t *nb = (uint8_t *)realloc(g->buf, ncap);
        if (!nb) return 0;  // signals error to curl
        g->buf = nb; g->cap = ncap;
    }
    memcpy(g->buf + g->len, ptr, n);
    g->len += n;
    return n;
}

static _Thread_local CURL *tls_h = NULL;

static void tls_reset(void) {
    if (tls_h) { curl_easy_cleanup(tls_h); tls_h = NULL; }
}

static CURL *tls_handle(const sftp_target *t) {
    if (tls_h) return tls_h;
    CURL *h = curl_easy_init();
    if (!h) return NULL;
    char userpwd[512];
    snprintf(userpwd, sizeof(userpwd), "%s:%s", t->user, t->pass);
    curl_easy_setopt(h, CURLOPT_USERPWD, userpwd);
    if (t->known_hosts)
        curl_easy_setopt(h, CURLOPT_SSH_KNOWNHOSTS, t->known_hosts);
    curl_easy_setopt(h, CURLOPT_SSH_AUTH_TYPES, CURLSSH_AUTH_PASSWORD);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, 120L);
    tls_h = h;
    return h;
}

int sftp_download(const sftp_target *t, const char *remote_path,
                  uint8_t **out, size_t *out_len) {
    *out = NULL; *out_len = 0;

    char url[2048];
    snprintf(url, sizeof(url), "sftp://%s:%d%s", t->host, t->port, remote_path);

    grow_buf g = {0};
    CURLcode res = CURLE_OK;
    for (int attempt = 0; attempt < 4; ++attempt) {
        CURL *h = tls_handle(t);
        if (!h) { free(g.buf); return -1; }
        curl_easy_setopt(h, CURLOPT_URL, url);
        curl_easy_setopt(h, CURLOPT_NOBODY, 0L);
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &g);
        g.len = 0;
        res = curl_easy_perform(h);
        if (res == CURLE_OK) break;
        if (res == CURLE_REMOTE_FILE_NOT_FOUND) break;  // don't retry a real 404
        fprintf(stderr, "sftp: download %s attempt %d failed: %s\n",
                remote_path, attempt + 1, curl_easy_strerror(res));
        tls_reset();  // connection may be dead; rebuild next attempt
    }
    if (res != CURLE_OK) { free(g.buf); return -1; }
    *out = g.buf; *out_len = g.len;
    return 0;
}

long sftp_stat_size(const sftp_target *t, const char *remote_path) {
    char url[2048];
    snprintf(url, sizeof(url), "sftp://%s:%d%s", t->host, t->port, remote_path);

    for (int attempt = 0; attempt < 3; ++attempt) {
        CURL *h = tls_handle(t);
        if (!h) return -1;
        curl_easy_setopt(h, CURLOPT_URL, url);
        curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
        CURLcode res = curl_easy_perform(h);
        curl_easy_setopt(h, CURLOPT_NOBODY, 0L);
        if (res == CURLE_OK) {
            curl_off_t len = -1;
            curl_easy_getinfo(h, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &len);
            return len >= 0 ? (long)len : -1;
        }
        if (res == CURLE_REMOTE_FILE_NOT_FOUND) return -1;
        tls_reset();
    }
    return -1;
}
