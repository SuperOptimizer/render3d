/* Missing moving data must be retried without editing the affine transform;
 * legitimate zero data must settle instead of periodically refilling. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "core/regvol.h"
#include "synthtree.h"
#include <assert.h>
#include <time.h>
int main(void) {
  char root[] = "/tmp/r3d_reg_retry_XXXXXX", path[256], saved[256];
  assert(mkdtemp(root));
  uint32_t dims[3] = {128, 128, 128};
  assert(st_make_tree(root, dims, 1, 0) == 0);
  snprintf(path, sizeof path, "%s/volcomp/L0/0_0_0.vcs", root);
  snprintf(saved, sizeof saved, "%s/saved.vcs", root);
  assert(rename(path, saved) == 0);
  r3d_regvol rv;
  assert(r3d_regvol_open(&rv, root, dims) == 0);
  uint8_t block[4096];
  uint32_t gen = r3d_regvol_blockgen(&rv, 0, 1, 1, 1);
  assert(gen);
  assert(!r3d_regvol_blockfetch_complete(&rv, 0, 1, 1, 1, block));
  for (size_t i = 0; i < sizeof block; i++)
    assert(block[i] == 0);
  assert(r3d_regvol_blockgen(&rv, 0, 1, 1, 1) == 0);
  assert(r3d_regvol_blockgen(&rv, 0, 2, 2, 2) ==
         gen); /* no global invalidation */
  assert(rename(saved, path) == 0);
  /* Retry cooldown is one second; appearance is detected despite the CPU
   * sampler's longer negative-cache lifetime, with unchanged transformgen. */
  struct timespec delay = {.tv_sec = 1, .tv_nsec = 50000000};
  nanosleep(&delay, NULL);
  assert(r3d_regvol_blockgen(&rv, 0, 1, 1, 1) == gen);
  assert(r3d_regvol_blockfetch_complete(&rv, 0, 1, 1, 1, block));
  bool nonzero = false;
  for (size_t i = 0; i < sizeof block; i++)
    nonzero |= block[i] != 0;
  assert(nonzero);
  for (uint32_t i = 0; i < 100; i++)
    assert(r3d_regvol_blockgen(&rv, 0, 1, 1, 1) == gen);
  r3d_regvol_bump(&rv);
  assert(r3d_regvol_blockgen(&rv, 0, 1, 1, 1) != gen);
  r3d_regvol_close(&rv);
  volcomp_shard_writer *writer = volcomp_shard_create(path, 1024, 128, 0, 2);
  assert(writer && volcomp_shard_put_zero(writer, 0) == 0 &&
         volcomp_shard_close(writer) == 0);
  assert(r3d_regvol_open(&rv, root, dims) == 0);
  gen = r3d_regvol_blockgen(&rv, 0, 1, 1, 1);
  assert(gen);
  assert(r3d_regvol_blockfetch_complete(&rv, 0, 1, 1, 1, block));
  assert(rv.retry == NULL && r3d_regvol_blockgen(&rv, 0, 1, 1, 1) == gen);
  for (size_t i = 0; i < sizeof block; i++)
    assert(block[i] == 0);
  assert(r3d_regvol_blockfetch_complete(&rv, 0, 10, 10, 10, block));
  assert(r3d_regvol_blockgen(&rv, 0, 10, 10, 10) == 0 && rv.retry == NULL);
  r3d_regvol_close(&rv);
  st_rm_tree(root, 1);
  puts("registration: unavailable blocks retry; known air and completed blocks "
       "stay stable");
  return 0;
}
