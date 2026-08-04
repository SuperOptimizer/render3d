// localio.c — see localio.h.

#include "localio.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

long local_size(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return (long)st.st_size;
}

int local_read(const char *path, uint8_t **out, size_t *out_len) {
    *out = NULL; *out_len = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return -1; }
    rewind(f);
    uint8_t *buf = (uint8_t *)malloc(sz ? (size_t)sz : 1);
    if (!buf) { fclose(f); return -1; }
    if (sz && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);
    *out = buf; *out_len = (size_t)sz;
    return 0;
}

static int mkdirs_for(const char *path) {
    char tmp[2048];
    snprintf(tmp, sizeof(tmp), "%s", path);
    char *slash = strrchr(tmp, '/');
    if (!slash) return 0;
    *slash = 0;  // parent dir of the file
    for (char *p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

int local_write_atomic(const char *path, const uint8_t *buf, size_t len) {
    if (mkdirs_for(path) != 0) {
        fprintf(stderr, "localio: mkdir -p for %s failed: %s\n", path, strerror(errno));
        return -1;
    }
    char tmp[2048];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
    FILE *f = fopen(tmp, "wb");
    if (!f) {
        fprintf(stderr, "localio: open %s failed: %s\n", tmp, strerror(errno));
        return -1;
    }
    if (len && fwrite(buf, 1, len, f) != len) {
        fprintf(stderr, "localio: short write %s: %s\n", tmp, strerror(errno));
        fclose(f); unlink(tmp);
        return -1;
    }
    if (fclose(f) != 0) { unlink(tmp); return -1; }
    if (rename(tmp, path) != 0) {
        fprintf(stderr, "localio: rename %s -> %s failed: %s\n", tmp, path, strerror(errno));
        unlink(tmp);
        return -1;
    }
    return 0;
}
