/* Direct GPU/CPU comparison of sparse block mip and occupancy batches. */
#include "vk/vkctx.h"
#include "vk/vkres.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BPA = 4, DIM = BPA * 16, ODIM = BPA * 2 };
static size_t index3(unsigned x, unsigned y, unsigned z, unsigned d) {
  return ((size_t)z * d + y) * d + x;
}
static unsigned clampi(int x, unsigned d) {
  return x < 0 ? 0u : (unsigned)x >= d ? d - 1u : (unsigned)x;
}
/* Integer round-to-nearest, ties-to-even: independent of shader float math. */
static uint8_t average8(unsigned sum) {
  unsigned q = sum / 8, r = sum % 8;
  return (uint8_t)(q + (r > 4 || (r == 4 && (q & 1u))));
}
static void barrier(VkCommandBuffer cmd, r3d_vkimage *im, unsigned first,
                    unsigned n) {
  r3d_vk_image_barrier(
      cmd, im->img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, first, n);
}
static void clear(r3d_vkctx *c, VkCommandPool pool, r3d_vkimage *im,
                  unsigned mips, unsigned value) {
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(c, pool);
  assert(cmd);
  r3d_vk_image_barrier(
      cmd, im->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT,
      VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, mips);
  VkClearColorValue col = {.float32 = {(float)value / 255.0f, 0, 0, 0}};
  VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
  vkCmdClearColorImage(cmd, im->img, VK_IMAGE_LAYOUT_GENERAL, &col, 1, &rr);
  barrier(cmd, im, 0, mips);
  assert(r3d_vk_oneshot_end(c, pool, cmd) == 0);
}
static void check_image(r3d_vkctx *c, VkCommandPool pool, r3d_vkimage *im,
                        unsigned mip, unsigned dim, const uint8_t *expected) {
  size_t size = (size_t)dim * dim * dim;
  r3d_vkbuf buf = {0};
  assert(r3d_vkbuf_create_host(c, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               &buf) == 0);
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(c, pool);
  assert(cmd);
  barrier(cmd, im, mip, 1);
  VkBufferImageCopy copy = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1},
      .imageExtent = {dim, dim, dim}};
  vkCmdCopyImageToBuffer(cmd, im->img, VK_IMAGE_LAYOUT_GENERAL, buf.buf, 1,
                         &copy);
  VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                         .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                         .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                         .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
  VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                          .memoryBarrierCount = 1,
                          .pMemoryBarriers = &mb};
  vkCmdPipelineBarrier2(cmd, &dep);
  assert(r3d_vk_oneshot_end(c, pool, cmd) == 0);
  const uint8_t *got = buf.mapped;
  for (size_t i = 0; i < size; i++) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "blockpost: mip %u dim %u index %zu got %u expected %u\n",
              mip, dim, i, got[i], expected[i]);
      abort();
    }
  }
  r3d_vkbuf_destroy(c, &buf);
}
static void reference(uint8_t *mips[4], uint8_t *raw, uint8_t *dil,
                      const uint32_t *pairs, unsigned n, unsigned pass) {
  for (unsigned i = 0; i < n; i++) {
    unsigned s = pairs[2 * i], b = pairs[2 * i + 1];
    unsigned sx = s % BPA, sy = (s / BPA) % BPA, sz = s / (BPA * BPA);
    for (unsigned z = 0; z < 16; z++)
      for (unsigned y = 0; y < 16; y++)
        for (unsigned x = 0; x < 16; x++)
          mips[0][index3(sx * 16 + x, sy * 16 + y, sz * 16 + z, DIM)] =
              (uint8_t)((x * 17 + y * 11 + z * 3 + s * 7) %
                        (pass && s == 9 ? 15u
                                        : 45u + (s * 13 + pass * 47) % 200u));
    for (unsigned m = 1; m < 4; m++) {
      unsigned edge = 16u >> m, d = DIM >> m;
      for (unsigned z = 0; z < edge; z++)
        for (unsigned y = 0; y < edge; y++)
          for (unsigned x = 0; x < edge; x++) {
            unsigned ox = sx * edge + x, oy = sy * edge + y, oz = sz * edge + z;
            unsigned sum = 0;
            for (unsigned dz = 0; dz < 2; dz++)
              for (unsigned dy = 0; dy < 2; dy++)
                for (unsigned dx = 0; dx < 2; dx++)
                  sum += mips[m - 1][index3(ox * 2 + dx, oy * 2 + dy,
                                            oz * 2 + dz, d * 2)];
            mips[m][index3(ox, oy, oz, d)] = average8(sum);
          }
    }
    for (unsigned z = 0; z < 2; z++)
      for (unsigned y = 0; y < 2; y++)
        for (unsigned x = 0; x < 2; x++) {
          uint8_t maximum = 0;
          for (unsigned dz = 0; dz < 8; dz++)
            for (unsigned dy = 0; dy < 8; dy++)
              for (unsigned dx = 0; dx < 8; dx++) {
                uint8_t v =
                    mips[0][index3(sx * 16 + x * 8 + dx, sy * 16 + y * 8 + dy,
                                   sz * 16 + z * 8 + dz, DIM)];
                if (v > maximum)
                  maximum = v;
              }
          raw[index3((b % BPA) * 2 + x, ((b / BPA) % BPA) * 2 + y,
                     (b / (BPA * BPA)) * 2 + z, ODIM)] = maximum;
        }
  }
  for (unsigned z = 0; z < ODIM; z++)
    for (unsigned y = 0; y < ODIM; y++)
      for (unsigned x = 0; x < ODIM; x++) {
        uint8_t maximum = 0;
        for (int dz = -1; dz <= 1; dz++)
          for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
              uint8_t v = raw[index3(clampi((int)x + dx, ODIM),
                                     clampi((int)y + dy, ODIM),
                                     clampi((int)z + dz, ODIM), ODIM)];
              if (v > maximum)
                maximum = v;
            }
        dil[index3(x, y, z, ODIM)] = maximum;
      }
}
int main(void) {
  r3d_vkctx c;
  if (r3d_vkctx_create(&c, NULL, 0, false) != 0)
    return 77;
  VkCommandPool pool;
  VkCommandPoolCreateInfo pi = {.sType =
                                    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                .queueFamilyIndex = c.qfam};
  assert(vkCreateCommandPool(c.dev, &pi, NULL, &pool) == VK_SUCCESS);
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  r3d_vkimage atlas, raw, dil;
  assert(r3d_vkimage_create(&c, VK_FORMAT_R8_UNORM, (VkExtent3D){DIM, DIM, DIM},
                            4, usage, &atlas) == 0);
  assert(r3d_vkimage_create(&c, VK_FORMAT_R8_UNORM,
                            (VkExtent3D){ODIM, ODIM, ODIM}, 1, usage,
                            &raw) == 0);
  assert(r3d_vkimage_create(&c, VK_FORMAT_R8_UNORM,
                            (VkExtent3D){ODIM, ODIM, ODIM}, 1, usage,
                            &dil) == 0);
  clear(&c, pool, &atlas, 4, 37);
  clear(&c, pool, &raw, 1, 0);
  clear(&c, pool, &dil, 1, 0);
  VkSampler sampler;
  VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                            .magFilter = VK_FILTER_NEAREST,
                            .minFilter = VK_FILTER_NEAREST,
                            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST};
  assert(vkCreateSampler(c.dev, &si, NULL, &sampler) == VK_SUCCESS);
  r3d_vkcomp kernel[3];
  VkDescriptorType mt[] = {
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
  VkDescriptorType ot[] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                           VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
  const char *names[] = {"blockmips", "occmax", "occdilate"};
  for (unsigned i = 0; i < 3; i++) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.spv", R3D_SPV_DIR, names[i]);
    assert(r3d_vkcomp_create(&c, path, i ? ot : mt, i ? 3 : 5, 12,
                             &kernel[i]) == 0);
  }
  VkImageView views[4];
  for (unsigned m = 0; m < 4; m++) {
    VkImageViewCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = atlas.img,
        .viewType = VK_IMAGE_VIEW_TYPE_3D,
        .format = VK_FORMAT_R8_UNORM,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 1}};
    assert(vkCreateImageView(c.dev, &vi, NULL, &views[m]) == VK_SUCCESS);
    r3d_vkcomp_bind_image(&c, &kernel[0], m, mt[m], views[m], VK_NULL_HANDLE,
                          VK_IMAGE_LAYOUT_GENERAL);
  }
  for (unsigned i = 1; i < 3; i++) {
    r3d_vkcomp_bind_image(&c, &kernel[i], 0, ot[0],
                          i == 1 ? atlas.view : raw.view, sampler,
                          VK_IMAGE_LAYOUT_GENERAL);
    r3d_vkcomp_bind_image(&c, &kernel[i], 1, ot[1],
                          i == 1 ? raw.view : dil.view, VK_NULL_HANDLE,
                          VK_IMAGE_LAYOUT_GENERAL);
  }
  r3d_vkbuf stage;
  size_t bytes = (size_t)DIM * DIM * DIM;
  assert(r3d_vkbuf_create_host(&c, bytes + 64,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               &stage) == 0);
  for (unsigned i = 0; i < 3; i++)
    r3d_vkcomp_bind_buffer(&c, &kernel[i], i ? 2 : 4, stage.buf, 0,
                           VK_WHOLE_SIZE);
  uint8_t *mips[4];
  for (unsigned m = 0; m < 4; m++) {
    size_t d = DIM >> m;
    mips[m] = malloc(d * d * d);
    assert(mips[m]);
    memset(mips[m], 37, d * d * d);
  }
  uint8_t expected_raw[ODIM * ODIM * ODIM] = {0},
                                     expected_dil[ODIM * ODIM * ODIM] = {0};
  const uint32_t batches[2][8] = {{9, 0, 2, 1, 61, 63, 0, 5},
                                  {9, 0, 44, 16, 1, 4, 0, 0}};
  for (unsigned pass = 0; pass < 2; pass++) {
    unsigned n = pass ? 3 : 4;
    reference(mips, expected_raw, expected_dil, batches[pass], n, pass);
    memcpy(stage.mapped, mips[0], bytes);
    memcpy((uint8_t *)stage.mapped + bytes, batches[pass], n * 8);
    VkCommandBuffer cmd = r3d_vk_oneshot_begin(&c, pool);
    assert(cmd);
    barrier(cmd, &atlas, 0, 4);
    VkBufferImageCopy copy = {
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {DIM, DIM, DIM}};
    vkCmdCopyBufferToImage(cmd, stage.buf, atlas.img, VK_IMAGE_LAYOUT_GENERAL,
                           1, &copy);
    barrier(cmd, &atlas, 0, 4);
    uint32_t pc[3] = {BPA, BPA, (uint32_t)(bytes / 4)};
    r3d_vkcomp_dispatch(cmd, &kernel[0], pc, sizeof pc, n, 1, 1);
    barrier(cmd, &atlas, 0, 4);
    barrier(cmd, &raw, 0, 1);
    r3d_vkcomp_dispatch(cmd, &kernel[1], pc, sizeof pc, n, 1, 1);
    barrier(cmd, &raw, 0, 1);
    barrier(cmd, &dil, 0, 1);
    r3d_vkcomp_dispatch(cmd, &kernel[2], pc, sizeof pc, n, 1, 1);
    barrier(cmd, &dil, 0, 1);
    assert(r3d_vk_oneshot_end(&c, pool, cmd) == 0);
    for (unsigned m = 0; m < 4; m++)
      check_image(&c, pool, &atlas, m, DIM >> m, mips[m]);
    check_image(&c, pool, &raw, 0, ODIM, expected_raw);
    check_image(&c, pool, &dil, 0, ODIM, expected_dil);
  }
  for (unsigned i = 0; i < 3; i++)
    r3d_vkcomp_destroy(&c, &kernel[i]);
  for (unsigned m = 0; m < 4; m++) {
    vkDestroyImageView(c.dev, views[m], NULL);
    free(mips[m]);
  }
  vkDestroySampler(c.dev, sampler, NULL);
  r3d_vkbuf_destroy(&c, &stage);
  r3d_vkimage_destroy(&c, &atlas);
  r3d_vkimage_destroy(&c, &raw);
  r3d_vkimage_destroy(&c, &dil);
  vkDestroyCommandPool(c.dev, pool, NULL);
  r3d_vkctx_destroy(&c);
  puts("blockpost: sparse mip levels, retained slots, world occupancy and "
       "overlapping halos match CPU");
  return 0;
}
