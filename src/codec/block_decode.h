/* Adapted from volume-compressor vf_decode_sub.
 * MIT License
 * 
 * Copyright (c) 2026 SuperOpt
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/* Entropy restart adapter for the pinned volume-compressor revision.
 * Token validation/order mirrors vf_decode_sub; only the requested block is
 * dequantized/transformed. tests/test_decode_restart.c checks public-API parity.
 * All saved bitreader pointers refer to owned compressed bytes below. */
#define R3D_RESTART_BYTES (64u*1024u)
#define R3D_RESTART_SLOTS 4u
typedef struct { vf_rdec rd; vf_bitr br; int64_t dc; } r3d_checkpoint;
typedef struct {
  uint32_t sub, valid;
  size_t n;
  uint8_t bytes[R3D_RESTART_BYTES];
  r3d_checkpoint at[VF_BLOCKS_PER_SUB];
} r3d_restart;

static volcomp_status r3d_decode_sub(const vf_parsed *p, uint32_t s, uint32_t only_block,
                                     uint8_t *dst_block, r3d_restart *restart) {
  const uint8_t *rb = restart ? restart->bytes : p->payload + p->off[s];
  vf_rdec rd;
  if (!vf_rdec_init(&rd, rb, p->tok_n[s])) return VOLCOMP_ERR_CORRUPT;
  vf_bitr br;
  vf_br_init(&br, rb + p->tok_n[s], p->byp_n[s]);
  int64_t prev_dc = 0;
  uint32_t first=s*VF_BLOCKS_PER_SUB;
  if (restart) {
    for (uint32_t i=only_block%VF_BLOCKS_PER_SUB; i>0; i--) {
      if (restart->valid & (1u<<i)) {
        rd=restart->at[i].rd; br=restart->at[i].br; prev_dc=restart->at[i].dc;
        first+=i; break;
      }
    }
  }
  float blk[VF_BLKV] = {0}, vox[VF_BLKV]; /* only the requested block writes coefficients */
  for (uint32_t bi = first; bi <= only_block; bi++) {
    int t = vf_rdec_get(&rd, &p->models[VF_DC_CTX]);
    if (t < 0 || (uint32_t)t > VF_TOKMAX_DC) return VOLCOMP_ERR_CORRUPT;
    uint32_t u;
    if (!vf_hyb_read(&br, (uint32_t)t, &u)) return VOLCOMP_ERR_CORRUPT;
    int64_t dc = prev_dc + vf_unzigzag(u);
    if (dc < -VF_DC_ABS_MAX || dc > VF_DC_ABS_MAX) return VOLCOMP_ERR_CORRUPT;
    prev_dc = dc;
    bool wanted=bi==only_block;
    if (wanted && dc != 0) {
      float mag = vf_dequant_dc((uint32_t)(dc < 0 ? -dc : dc), p->step[0]) * 0.015625f;
      blk[0] = dc < 0 ? -mag : mag;
    }
    uint32_t pos = 1, nnz_ac = 0, zmask = 1u, hmask = 0;
    for (;;) {
      t = vf_rdec_get(&rd, &p->models[vf_run_ctx(pos)]);
      if (t < 0) return VOLCOMP_ERR_CORRUPT;
      if ((uint32_t)t == VF_TOK_EOB) break;
      if ((uint32_t)t > VF_TOKMAX_RUN) return VOLCOMP_ERR_CORRUPT;
      uint32_t run;
      if (!vf_hyb_read(&br, (uint32_t)t, &run)) return VOLCOMP_ERR_CORRUPT;
      pos += run;
      if (pos >= VF_BLKV) return VOLCOMP_ERR_CORRUPT;
      int lt = vf_rdec_get(&rd, &p->models[vf_level_ctx(pos, run)]);
      if (lt < 0 || (uint32_t)lt > VF_TOKMAX_LVL) return VOLCOMP_ERR_CORRUPT;
      uint32_t mag1, sign;
      if (!vf_hyb_read(&br, (uint32_t)lt, &mag1) || !vf_br_get(&br, 1, &sign))
        return VOLCOMP_ERR_CORRUPT;
      if (mag1 + 1u > VF_AC_MAG_MAX) return VOLCOMP_ERR_CORRUPT;
      if (wanted) {
        uint32_t nat = VF_SCAN16[pos];
        float stp = p->step[vf_radius(nat)];
        float v = vf_dequant_ac(mag1 + 1u, stp) * vf_dct_scale(nat);
        blk[nat] = sign ? -v : v;
        zmask |= 1u << (nat >> 8);
        hmask |= (nat >> 7 & 1u) << (nat >> 8);
        nnz_ac++;
      }
      pos++;
    }
    uint32_t next=bi%VF_BLOCKS_PER_SUB+1u;
    if (restart && next<VF_BLOCKS_PER_SUB) {
      restart->at[next]=(r3d_checkpoint){rd,br,prev_dc};
      restart->valid|=1u<<next;
    }
    if (!wanted) continue;
    if (nnz_ac==0) memset(dst_block,vf_flat_value(blk[0]),VF_BLKV);
    else {
      vf_dct16_inv(blk,zmask,hmask,vox);
      vf_scatter_block(dst_block,256,16,vox);
    }
    return VOLCOMP_OK;
  }
  return VOLCOMP_ERR_ARG;
}
