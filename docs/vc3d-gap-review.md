# vc3d vs render3d tracer — gap review

Scope: `D:/render3d/src/core/tracer.c` (4732 lines) + `tracer.h` against vc3d's
`GrowPatch.cpp` / `GrowSurface.cpp` / `CostFunctions.hpp` / `NormalGridVolume.cpp` /
`vc_grow_seg_from_seed.cpp` / `vc_tifxyz_winding.cpp` / diffusion spiral service.
Every claim below was re-checked against our source; claims that turned out to be
already implemented, or overstated, are listed in §3.

## 1. Executive summary

Our tracer reproduces vc3d's *inner loop* faithfully — seed quad, per-candidate
place → 50-iteration LM → radius-1 → radius-3, the same loss family (DistLoss,
StraightLoss, SymmetricDirichlet, space-line, normal-constraint-plane), the same
thread separation rule — and adds machinery vc3d does not have (umbilicus winding
prior, self-overlap hinge, winding-potential field, ribbon multi-wrap seeding).

What is missing is almost entirely the *outer* machinery: the parts of vc3d that
decide **whether a point is allowed to exist**, **how to undo a decision**, and
**how to measure the result**.

Four structural themes:

1. **Everything we commit is permanent.** No pre-solve veto, no per-cell
   generation stamp, no rewind, no retry (`R3D_TR_PROC` = offered once, ever),
   no consensus gate in fusion, no delete pass. vc3d rejects cheaply *before*
   solving, retries under an annealed threshold, and can roll back to a
   generation or a best snapshot.
2. **Our data-term gradient contains forces vc3d deliberately removes.** The
   forward difference in `ncp_residual4` differentiates through 1/d² weights,
   ROI membership, the angular weight and the snap target — all of which vc3d
   freezes with `.val()`. Plus we emit a constant snap penalty where vc3d
   returns exactly 0 (no coverage).
3. **The sheet-gap / winding model is a single global scalar assigned once.**
   `sp_om_meas` is one median driving the hinge radius, the spacing target, the
   winding normaliser and the hash cell size; winding is assigned at placement
   and never re-derived even though every cell is subsequently moved by three
   later solve passes; the spacing residual is unsigned, so a wrap that has
   flipped to the wrong side is held at perfect spacing on the wrong side.
4. **We measure almost nothing.** meta.json carries format/type/scale/donor
   count. No area, no bbox, no hole fraction, no donor agreement, no winding
   consistency, no slant. QC is folds/kinks/twist only — none of which can see
   a wrong-wrap capture or a coherent shear.

## 2. Confirmed gaps, ranked by expected trace-quality impact

### G1. Rejected candidates are placed, solved, and drag their neighbourhood before being failed
**Ours** `tracer.c:3326-3331` commits `R3D_TR_SET` and `nset++`; `:3339`
runs the 50-iteration placement solve; `:3351-3352` radius-1 + radius-3 local
opts; only at `:3361/:3395-3401` is the cell flipped to `R3D_TR_FAIL`. Nothing
restores the disc that was solved with the bogus point as a fixed neighbour.
Our box is z-only (`cfg.z_min/z_max`, `tracer.h:42`).
**vc3d** `GrowPatch.cpp:4851-4855` tests the mean of the candidate's valid 3×3
neighbours against x/y/z min/max and `continue`s *before* `dpoints(p)` is
written and before `state(p)` is set; the normal loss itself is z-gated
(`CostFunctions.hpp:1167-1169`).
**Why it matters** On z-clamped/ribbon/CT-bounded runs the front sits on the
clamp for many generations, so a large fraction of candidates are
place-solve-retract cycles. Each leaves a permanent scar: surviving edge cells
were pulled toward a point that no longer exists, systematically biased toward
the dead region. Also pure wasted work (50 LM iters + 5 sweeps per doomed cell).
**Port** In `tr_place_cand` before `state[k]=SET`: compute the valid-3×3 mean and
test against z/x/y limits and `t->vdim`; return false without touching pos/state.
Add `x_min/x_max/y_min/y_max` to `r3d_tracer_cfg`. Move the ribbon CT/DT test
(`:4045-4080`, currently a whole generation late) into the same pre-veto on the
parent-average position. Keep the post-solve test as a second line of defence but
snapshot the radius-3 disc before the placement solve and `memcpy` it back on
failure. ~80 lines, no new data structures.

### G2. The normal-grid gradient carries three spurious force components
**Ours** `ncp_residual4` (`tracer.c:1155-1306`) forward-differences the *whole*
residual: `aw[v] = 0.5*(1-na*na)` is recomputed per probe (`:1217`), the
straddle corner `Bn` is re-chosen per probe (`:1176-1197`), the segment loop
recomputes `d2`/ROI accept/`1/dd` per probe (`:1268-1276`), and `best_snap[v]`
is re-selected per probe (`:1279-1298`).
**vc3d** autodiffs but freezes every configuration-selecting quantity on the
scalar part: `dist_sq` from `.val()`ed points (`CostFunctions.hpp:1339-1345`),
ROI reject on `.val()`ed points (`:1332-1337`), `angle_weight` a plain `double`
(`:1218-1226`), snap target cached on `.val()`ed endpoints (`:1375-1441`). Only
`dot(edge_normal, seg_normal)` and the distance to a *frozen* segment carry
derivatives.
**Why it matters** (a) differentiating Σw(1−dot)/Σw through w=1/d² is a
translational pull toward whichever nearby polyline is best aligned — i.e. a
direct wrap-jump mechanism, since the neighbouring sheet is usually well
aligned; (b) differentiating `aw` is a torque rotating quads perpendicular to
each Cartesian plane, summed incoherently over 3 planes × 4 rotations × 4 quads;
(c) ROI/snap-target identity flipping between base and probe injects step
discontinuities into the Jacobian exactly where the data term is strongest.
**Port** Compute the configuration once from variant 0 and reuse it: hoist `aw`;
freeze `Bn`/slice; compute `dmx/dmy/d2/dd` and the ROI accept from `vmx[0]`;
run the snap search only for v=0 and evaluate `d2a/d2e` for v=1..3 against the
frozen segment endpoints. Mirror in the scalar `ncp_residual` (`:1047-1148`).

### G3. Empty / missing normal-grid neighbourhood emits a constant penalty instead of zero
**Ours** `tracer.c:1305` — `snap_loss = best_snap[v] >= 0 ? … : 1.0`, so a cell
with `g==NULL`, `g->empty`, or `hd->nseg==0` yields `out[v] = 0.1*aw[v]`.
**vc3d** `CostFunctions.hpp:1325-1327` returns `T(0)` *before* the snapping
section when `nearby_paths.empty()`; `:1176-1178` returns early when the slice
file is missing.
**Why it matters** In every uncovered region — outside the scanned extent,
across damage, on any missing `.grid` — the residual becomes a pure function of
quad orientation through `aw`, i.e. a systematic torque flattening quads into
the Cartesian planes, with nothing opposing it. Fires precisely where the tracer
has no evidence and should be governed by DIST/STRAIGHT/SDIR alone.
**Port** Return 0 immediately when `!hd || hd->nseg==0` or `!g || g->empty`;
track per-variant `nseen[v]` of ROI-accepted segments and apply the 1.0
no-target penalty only when `nseen[v] > 0`. Do G2's ROI freeze first so `nseen`
is variant-independent. ~15 lines; pairs naturally with G2.

### G4. Winding is assigned at placement and never re-derived; no winding consistency check
**Ours** `tr_wind_assign` is called exactly twice, both inside `tr_place_cand`
(`:3331` preliminary, `:3355` after the placement solve). Every cell is
subsequently moved by radius-3 opts, the every-8th-generation global solve
(`:4103-4131`) and the refine pass — the winding value never follows. That
stale value gates the self-overlap hinge (`:2231`), the two-sided spacing pull
(`:2270`), the donor pull's winding gate (`:3099`), the donor adoption gate
(`:3344`), the sibling-wrap stop (`:3383`), the spiral fit and `tr_spiral_flag`.
QC (`tr_qc`, `:3640-3694`) measures folds/kinks/twist only.
**vc3d** `vc_tifxyz_winding.cpp:429-556` relaxes winding as a *field*: propagate
across normal-ray-linked column pairs, diffuse over a multi-scale {1,4,16}
neighbourhood at the local rate, anneal `rough_w` 1→0, reject links whose column
gap deviates >1/3 from the local reference (`:452`), and emit
`winding_err = |1 − Δw·wind_x_ref|` as a per-cell QC map (`:566-576`).
Winding itself is recovered from geometry (normal-ray self-intersections,
`:104-205`), with a `two_halves` test rejecting hits on the opposite limb.
**Why it matters** One cell dragged across a wrap boundary by a later solve
keeps a stale winding forever, and that single value blinds five separate
wrong-wrap defences at once — the hinge skips it (`|Δw| < TR_SELF_DW`), the
spacing pull targets it, the donor gate admits it. Causal accumulation is also
path-dependent: around a closed ribbon loop the accumulated winding need not
return to an integer and nothing detects or distributes the closure error.
**Port** (a) `tr_wind_relax(t, iters)` called from the same slot as
`tr_spiral_fit`: Jacobi over SET cells, `w[k] = mean over 4-neighbours of
(w[n] + Δθ/2π)` with the seed (and each ribbon block's seed) pinned; reject
edges with `|Δθ/2π| > 0.25`; 20-40 sweeps. (b) `werr = |w − relaxed|` per cell;
clamp conf to 0.25 where `werr > 0.3` — an intrinsic wrong-wrap detector that
works with no spiral model and no winding-potential field. (c) A geometry-only
second opinion `tr_qc_wrap` using the existing `tr_sfx` hash: ray-march ±normal,
collect self-crossings, take the per-band median column spacing, report
wrap-jump fraction and `wrap_err` p95.

### G5. The sheet gap is one global scalar, and the spacing residual is unsigned
**Ours** `tr_omega_measure` (`:2112-2143`) returns a single median over ≤512
samples; `tr_om_eff` feeds the hinge radius `0.55*om` (`:2214`), the two-sided
target `(d−om)/om` (`:2276`), the winding-prior normaliser, the `sfx` hash cell
size and the spiral flag threshold. `tr_res_self` (`:2209-2293`) is entirely
unsigned: `dw = fabs(w[k] − wself)`, and the pull drives `|d| → om` regardless
of side.
**vc3d** never uses one number: per-x-band median column spacing with a
*local* outlier gate (`vc_tifxyz_winding.cpp:354-393, :452`); in the spiral
service each point's `dists_low/dists_high` are Ceres parameter blocks
(`spiral.cpp:247-279`) tied by a smoothness residual, floored by
`MinDistanceConstraint`, anchored to their own prior under Huber. And the
distance is **signed**: `sign = (w_x*v_y − w_y*v_x) > 0 ? 1 : −1`
(`spiral_ceres.hpp:19-56`), with opposite-sign minimum constraints on low/high
(`spiral.cpp:330-332`) enforcing the radial *order* of the neighbouring wraps.
**Why it matters** Real cross-sections vary 2–3× in gap. Where the true gap is
smaller than our median the hinge shoves genuine adjacent wraps apart and the
spacing pull drags cells onto the wrong sheet; where larger, wraps interpenetrate.
Worse, unsigned spacing means a cell that has slipped *past* its inner neighbour
— the exact failure the self-overlap term exists to catch — is held at perfect
spacing on the wrong side and looks locally ideal to every residual we have.
**Port** (a) Replace `sp_om_meas` with a coarse (32-voxel-cell) `sp_om_field`
filled from the per-sample gaps already collected in `tr_omega_measure`,
nearest-filled and box-smoothed, with the global median as fallback; route every
`tr_om_eff` call site through `tr_om_at(t, p)`. (b) Cast the gap rays along the
local grid normal (the cross product the anti-fold hinge already builds) rather
than the umbilicus radial — a radial ray on a pancaked section overestimates the
gap by 1/cos, often 1.5–2×. (c) Sign the spacing residual: take the outward
reference from `grad(r1)` of the winding field (or the umbilicus radial),
`s = sign(dot(x−Q, ref)) * sign(w_self − w_k)`, residual `w2*(s*d − om)/om` —
identical when the ordering is right, ≈−2 when inverted; widen or drop the
Cauchy on the inverted branch so the repair force is not robustified away.

### G6. Fusion commits every candidate: no consensus gate, no annealed threshold, no retry
**Ours** `tr_don_support` is called once per placement (`:3356-3359`) and its
result is used *only* as a confidence floor (`:3191-3193`) and a wrong-wrap veto
in `tr_spiral_flag` (`:1667, :1692`). `tr_place_cand` has no cost or consensus
gate. `:4010` sets `R3D_TR_PROC` — offered once, ever — so a cell rejected now
is never re-offered. Candidates are consumed in fringe/raster order (`:4001-4012`).
**vc3d** `GrowSurface.cpp:2186-2214` per-candidate inlier vote (needs ≥2
independent surfaces agreeing within `same_surface_th` with `local_cost <
local_cost_inl_th`); `:2935` is the only commit path; `:3021-3030` rejection sets
`state(p)=0` — retryable; `:3050-3070` anneals `curr_best_inl_th` from 20 down
toward 10 whenever the fringe empties, then reseeds the whole fringe;
`:3300-3330` emergency retry at 2. Candidates are ordered by support depth /
neighbour count first (`CandidateOrdering.cpp:20-33`).
**Why it matters** This is the entire quality mechanism of fusion growth.
Without it a cell no donor supports, sitting on a weak DT ridge, becomes
permanent geometry and the parent of the next generation — errors propagate
instead of being deferred. We also cannot express "take the certain territory
first": the front advances in whatever order the raster happens to produce, which
is exactly the nondeterminism the every-8th-generation global solve is currently
paid to anneal away (see the comment at `:4098-4103`).
**Port** (1) *Ordering, standalone and cheap*: after filling `cands[]`, sort by
(axis run-depth desc, SET-8-neighbour count desc, index asc) before handing to
the pool — the pool's ≥7-cell separation logic consumes the array in order, so
sorting alone changes the schedule. (2) *Gate*: add a `tr_eval_cell` (0 LM
iterations, returns cost/nres); score = supporting-donor residual count; keep
`t->inl_th` on the tracer, initialised 20; below threshold revert to
`R3D_TR_EMPTY` (retryable), not PROC/FAIL. At a generation boundary with
`nnew==0` and `inl_th > 10`, drop by 2 and rebuild the fringe from all boundary
SET cells instead of terminating. Guard on `t->don != NULL` so raw tracing keeps
GrowPatch semantics. (3) *L-shape rule* (`GrowSurface.cpp:2391-2420`): reject a
candidate that closes no 2×2 block, except in the first ~30 generations — cheap,
kills the 1-cell spurs the anti-fold term currently has to fight.

### G7. No seed validation or snapping
**Ours** `tracer.c:3840-3846` only probes *coarser pyramid levels* until the DT
at the seed reads < 64; the seed itself is never moved and never rejected.
`src/main.c:3065` passes the camera focus verbatim.
**vc3d** `vc_grow_seg_from_seed.cpp:523-566` — reject chunks with no source data,
then march a random unit direction up to 128 voxels and take the first sample
with prediction ≥ 128 as the origin; `:508-519` prints the sampled seed value.
**Why it matters** The GUI seed is wherever the camera focus happens to be —
often a voxel or two off mid-sheet, sometimes in inter-sheet air. The first 2×2
quad then inflates in empty space and the first ~10 generations of global solves
lock in a sheet choice made from an arbitrary start. Cheapest quality lever in
the pipeline.
**Port** `tr_seed_snap(t, dt)` just before the seed-quad construction (~`:3836`):
if `td_tri(dt, cfg.seed)` > ~1.5, reuse the radial local-minimum scan already
written at `:3860-3874` (walk s ∈ [−40,+40] along the umbilicus radial, or ±x/±y/±z
without an umbilicus) and take the nearest minimum with value < 2.5. Fail
`r3d_tracer_start` with a distinct code when none is found, so the GUI can say
"seed is not on a sheet" instead of growing garbage. Write the snapped position
back to `t->cfg` for display.

### G8. Holes are filled by a Laplacian membrane, with no data term and no interiority test
**Ours** `r3d_tracer_save(..., fill)` (`:4464-4505`) runs 64 Jacobi iterations of a
4-neighbour average over every `state==SET && !keep` cell — no solve, no
prediction/normal-grid term, no topology test beyond `na >= 2`. The tracer never
revisits a hole during growth.
**vc3d** `GrowPatch.cpp:2766-2812` — `masked_blur` seed (`:2573-2627`), hole cells
marked valid, boundary pinned, then the *real* loss stack run over the ROI with
SNAP ramped 0 → 0.001 → 0.01 → 0.1 → default. Interiority is enforced:
`findContours` RETR_CCOMP keeping only contours with a parent (`:3616-3641,
:4541-4570`), a 4-cell margin, and a re-check that the ROI's outer 2-cell ring is
entirely known (`:2769-2778`).
**Why it matters** A membrane across a hole is smooth but data-blind — in a
scroll it very often cuts through another wrap, and the result is written into
x/y/z indistinguishable from traced points. More specifically: our fill re-seats
*tear-cut* cells too, i.e. cells the tear mask deliberately removed as
wrong-wrap captures, so `fill=true` partially undoes the safety check written
right above it. Filled cells also shrink toward the boundary centroid, so quads
inside a filled region are undersized and rendered ink is locally compressed.
**Port** (a) Gate: 8-connected component-label the to-fill set, flood from the
grid border to mark the exterior, and fill only components untouched by that
flood whose dilated bbox is inside the grid with an all-`keep` 2-cell ring.
(b) Solve: keep the membrane as the initial guess (it is exactly `masked_blur`'s
role), then run `tr_local_opt` over the hole cells only — first with geometry
terms only, then with the data term scaled 0.01 / 0.1 / 1.0 (see G9), then full
`TRF_ALL`. Recompute conf afterwards so a hole that found no polyline stays
below the save cutoff. Pinning is already expressible: push only hole cells into
the item list.

### G9. No staged weight relaxation on corrections, inpaint, or refine
**Ours** `r3d_tracer_refine` (`:4145-4160`) runs three *identical*
`tr_local_opt(radius 8, 6 sweeps)` passes at full weights. `NCP_W_SNAP` is baked
into `ncp_residual/4` as a compile-time constant (`:1024`).
**vc3d** `GrowPatch.cpp:4296-4318` — three passes around each correction at
radius `8 + ceil(spread)`: SNAP×0, DIST×0.3, STRAIGHT×0.1; then snap restored;
then full. Same continuation schedule in inpaint (`:2799-2810`) and
`resample_inside_boundary` (`:2672-2682`).
**Why it matters** A correction that asks the sheet to move to a neighbouring
wrap must cross a barrier: the snap term holds it on its current polyline and
DIST/STRAIGHT resist the transition stretch. At full weights our refine converges
back into the wrong-sheet basin — precisely the failure anchors exist to fix.
Cheapest possible fix; requires no new loss.
**Port** Add `double wdist, wstraight, wsnap` to the solve context, read by
`tr_res_dist`/`tr_res_straight`/`ncp_residual4` instead of the constants; run
refine pass 0 at {0.3, 0.1, 0}, pass 1 at {0.3, 0.1, 1.0}, pass 2 at all 1.0,
and widen the radius to `8 + anchor spread`. Same schedule reused by G8.

### G10. User corrections are a hard per-axis pull on one vertex, and no region is ever reopened
**Ours** `tr_eval` (`:3107-3117`) emits three axis-aligned residuals
`TR_W_ANC*(x[a] − anchor[a])` on exactly one owned cell, weight 2.0, owner picked
by nearest-cell search within 3 grid steps (`:3701-3733`). `r3d_tracer_refine`
frees a radius-8 disc with *no positional memory at all*.
**vc3d** three separate mechanisms we lack: (a) `PointsCorrectionLoss`
(`CostFunctions.hpp:1615-1728`) is a **plane** distance — for each of the quad's
4 corners, the signed distance of the target to the plane spanned there, weighted
by `max(0, 1 − dist/40)` in 3D and `max(0, 1 − grid_dist/2)` in uv, added on 4
quads per point; (b) `NormalOnlyPenalty` + `NormalDisplacementClamp`
(`:1123-1193`, weight 10) — during re-optimisation each point remembers where it
was *tangentially only*, free to move along its own normal, with a softplus cap
of 3 voxels on normal motion at the region boundary; (c) corrections define a
**region**: an approval mask flood-filled from the correction point, reopened and
regrown against a frozen ring (`GrowPatch.cpp:4060-4130`).
**Why it matters** Pulling one vertex onto an arbitrary world point fights
DistLoss/StraightLoss/SymmetricDirichlet directly: the anneal either wins and
leaves a dimple/parameterisation defect, or loses and the anchor never engages
(our `anchor N final distance` log line exists precisely because this happens).
vc3d's plane form lets the sheet slide tangentially at zero cost, which is right
— a correction is about *which sheet*, not *which uv*. And with nothing pinning
the parameterisation, a 6-sweep radius-8 anneal can translate/rotate the whole
disc along the sheet, silently breaking uv continuity with the untouched region.
Where the patch jumped a wrap, the correct geometry is not a perturbation of the
wrong geometry at all — discard-and-regrow is the only clean fix.
**Port** In order: (1) tangential-only reopt anchor — store `anchor_pos` +
`anchor_nrm` at the start of a refine pass, emit `w*(d − (d·n)n)` with w=10 plus a
softplus barrier on `|d·n| > 3` at the disc edge, under a new `TRF_REOPT` flag
enabled only during refine/inpaint. (2) Quad-plane correction loss replacing the
three axis residuals, with the 2-cell/40-voxel falloffs; drop `TR_W_ANC` to 1.0
once the term is tangential-free. (3) `r3d_tracer_reopt(t, seed_cell, radius)`
— flood-fill from the anchor's cell over low-conf cells, abort if it reaches the
border, mark the interior EMPTY with a frozen boundary ring, regrow restricted to
the mask (needs the growth mask from G13).

### G11. No per-cell generation stamp: no rewind, no resume from a saved segment
**Ours** only a scalar `gens_done`/`ring` (`tracer.h:116-118`); grep for
"generations" in tracer.c finds two comments. `r3d_tracer_save` writes x/y/z +
winding + meta. `r3d_tracer_grow` (`:4287-4330`) only re-expands the *in-memory*
grid of the current session; nothing loads a tifxyz back into the tracer.
**vc3d** `GrowPatch.cpp:3753` `generations` mat, stamped at every placement
(`:4780, :4887`), saved as a channel (`:3778`), synthesized for legacy surfaces
(`:3496-3506`); `--resume/--rewind-gen/--resume-opt` (`vc_grow_seg_from_seed.cpp:161-178`);
`:4030-4045` drops every point with `gen > start_gen` — the actual rewind — and
regrows; `:4444-4470` resume-opt skip/local/global.
**Why it matters** Our only recovery from a trace that went wrong at generation
40 of 120 is to throw it away. Rewind-past-the-jump, drop an anchor, regrow is
the standard production fix; without a stamp we also cannot bound a re-solve to
"the recently placed, least-settled cells". Our saved tifxyz is also not
rewindable by vc3d tooling.
**Port** `uint16_t *gen_of` alongside state, = 1 for the seed quad, = generation
in `tr_place_cand`, realloc'd in `r3d_tracer_grow`; written as `generations.tif`;
`r3d_tracer_load(t, dir, cfg)` mapping planes back into pos/state/conf/wind/gen_of
with the existing centering arithmetic; `r3d_tracer_rewind(t, gen)` setting
cells with `gen_of > gen` to EMPTY. Resume then flows through the fringe rebuild
already at `:3971-3990`. Expose `rewind_gen` in the tracer panel.

### G12. Fusion is nearest-3D-point only: no per-cell donor uv, no affine prediction, no high-res resample
**Ours** `tr_don_closest` returns one nearest 3D point from one donor
(`:1828`); the fusion residual (`:3088-3105`) is a single 3-component pull;
adoption (`:3336-3348`) hard-snaps to that point; there is no per-cell donor id
or uv anywhere in `r3d_tracer`. Output is the coarse grid at `scale = 1/step`
(`:4533-4540`).
**vc3d** every locally-known surface within `same_surface_th` becomes a *member*
of the cell with its own uv (`GrowSurface.cpp:2960-3005`), each contributing a
SurfaceLossD; uv carries its own residual family (DistLoss2D/StraightLoss2D,
`:456-500, :669-694`) and `local_cost` is evaluated in uv space (`:806-836`);
candidate initial guesses are **affine uv extrapolations** with the worst sample
trimmed (`:203-256, :2789-2812`); and the final surface is resampled through the
donors' own parameterisation at `step×` resolution with per-sample outlier
rejection and multi-donor averaging (`surftrack_genpoints_hr`, `:861-940,
:3540-3546`).
**Why it matters** (a) Nearest-3D-point matching picks the wrong fold of the
*same* donor where a patch folds back or two of its wraps pass close — our
winding gate does not catch that (both folds have similar winding) and is skipped
entirely for donors without `winding.tif`. (b) Nearest-point is order-0 and
direction-blind: at a fold or overlap the nearest point can be *behind* the
front, so the grid stalls or doubles back. (c) The whole point of fusing existing
segments is to inherit their fine geometry (traced at src_step 20 with their own
optimisation) — emitting only the coarse quad grid throws that away and discards
the multi-donor averaging that suppresses per-donor noise.
**Port** In order: uv storage (`int8_t *dcell_id; float *dcell_uv;` for ≤4 slots,
allocated next to `dsup`) → affine uv initial guess in `tr_place_cand` (3×3 normal
equations, trim worst of ≥4 samples, fall back to parent+jitter with <2 samples)
→ one pull per occupied slot toward `donor_bilerp(di, uv)` → hr resample stage in
`r3d_tracer_save` writing at `1/src_step`. The wrong-fold test falls out for free:
reject a slot whose extrapolated uv and nearest-point uv disagree by >~2 grid units.

### G13. No steering: no growth-direction subset, no growth mask, no x/y box
**Ours** `:3990-4020` offers all 8 neighbours unconditionally; the only shaping
is the border margin `mv`, which ribbons set to 0.
**vc3d** `parse_growth_directions` (`GrowPatch.cpp:1341-1435`) restricts the
neighbour set; `allowed_growth_mask` rejects candidates per cell
(`:4694-4705`), imported from the resume surface (`:4358-4372`); plus the full
xyz box of G1.
**Why it matters** Ribbon mode is exactly what this exists for: the grid is
`(2*max_ring+10) × rib_rows` and growth should run along u with v fixed by the
row's z plane, yet every generation also offers ±v and diagonals, which the
z-anchor residual (`:3095-3105`) must then fight, and in multi-wrap mode a
diagonal candidate is the most likely way to leak into a neighbouring block's
rows. It is also the operator's steering wheel after a correction: regrow
outward only, or disable the side that keeps leaking.
**Port** `uint8_t grow_dirs` bitmask in cfg (0 = all), filtered in the candidate
loop; default to {+u, −u} when `cfg.rib_rows` is set. Optional `uint8_t
*grow_mask` allocated only when a resumed surface supplies one — also the
substrate for G10's region regrow.

### G14. Normal-grid store metadata is under-parsed: `sparse-volume` ignored, multiscale unsupported
**Ours** `ng_open` (`:548-556`) parses only `"spiral-step"`. Slice index is
`llround(A[axis])` (`:1075, :1218`); path is `%s/%s/%06d.grid` (`:751`);
prefetch strides by ±1 (`:755-757`). Level mismatch is only warned about
(`:3813`).
**vc3d** `NormalGridVolume.cpp:100-108, 312-318` snaps
`round(coord/sparse_volume)*sparse_volume`; strided prefetch (`:253-266`);
`normal-grid-multiscale` format with per-level metadata, `coordinate_scale`, a
level directory in the path, and `outputSpiralStep()` defining the grid step
(`:76-108, :160-170`; `CostFunctions.hpp:1152-1159`; `GrowPatch.cpp:3432-3453`).
**Why it matters** Against an N=4 store three of every four queries hit a
nonexistent file, cached as "known missing". And because `TRF_SPACE` is
deliberately disabled whenever grids are active (`tracer.c:3024-3025`), those
cells trace with **no data term at all** — plus, until G3 lands, a constant
spurious snap penalty. Against a published multiscale store our path template
resolves to nothing and the data term dies silently. Both are silent-failure
modes waiting for the next published store.
**Port** Parse `sparse-volume` into `ng_vol` (default 1) and route every slice
index through `ng_snap()`; stride prefetch by `d*sparse`. Parse
`format == "normal-grid-multiscale"`, clamp the level, load
`metadata.level<N>.json`, insert the level into the path, scale `a2[]/e2[]` by
`cscale` (leaving NCP_ROI2/QUERY_R/SNAP_* in grid pixels as vc3d does), and make
the store's spiral-step authoritative for `cfg.step` rather than a warning.

### G15. Nothing is measured: no area, no bbox, no hole fraction, no donor agreement, no min-area gate, no trim
**Ours** meta.json is format/type/scale/source/donor_segments (`:4536-4548`);
grep for `area` in tracer.c finds nothing; QC is folds/kinks/twist.
`r3d_tracer_save` writes the full `W×H` grid, and the grid is `2*max_ring+50`
columns wide, so an early-stopped or asymmetric run ships a large all-invalid
margin.
**vc3d** `SurfaceArea.hpp:47-115` per-quad two-triangle area; `area_vx2`,
`area_cm2`, `max_gen`, `elapsed_time_s` in the segment meta
(`GrowPatch.cpp:3730-3733, 3839-3847`); completeness metrics — valid bbox, fill
fraction, enclosed-hole fraction (`vc_grow_seg_from_segments.cpp:684-745`);
symmetric result↔reference distance stats with coverage fractions (`:619-670,
774-796`); a hard `min_area_cm` = 0.3 gate that **deletes the segment directory**
(`vc_grow_seg_from_seed.cpp:1440-1460`); `vc_tifxyz_trim.cpp:84-133` crops to the
valid bbox across all channels.
**Why it matters** Cell count is not surface area — a trace that stretched or
collapsed its quads has the same `nset` with wildly different physical coverage,
so we cannot say whether a run got more scroll than the last one. In fused runs
the donors are free ground truth: a trace that drifted a wrap away from its
donors over part of the patch currently produces *no signal at all*. And without
a min-area gate a seed that landed in noise leaves a plausible tifxyz behind that
later gets loaded as a donor. The enclosed-hole fraction is also the natural
driver for G8.
**Port** One pass computing quad area (same loop shape as the twist term at
`:3669-3690`), valid bbox + fill fraction, and enclosed-hole pixels from the same
border flood-fill G8 needs. `tr_qc_donor`: sample ~2000 kept cells, nearest donor
point via the existing index, mean/rms/p95/max + within-tolerance; reverse
direction via `tr_sfx` for donor coverage. `tr_qc_slant`: `tau = (e_u·e_v)/|e_u|²`
from the central-difference tangents already available in `tr_qc` — 25 lines, and
the only thing that can see a coherent shear that SymmetricDirichlet let build up.
Emit all of it into meta.json + the QC log line; refuse to save below min area;
crop to the valid bbox recording `grid_offset`.

### G16. Segment-level orchestration is absent
Lower individually, but together they are the difference between "one patch per
click" and scroll-wide coverage.
- **Ray-cast seed generator** (`SeedingWidget.cpp:864-1160`): one seed per sheet
  crossing along rays from a focus, each pushed to the intensity-run position
  maximising the *raw-CT* slice distance transform (mid-papyrus, not sheet edge).
  We have the 1D EDT (`td_edt1d`, `:86`) and the crossing scan already.
- **Expansion mode** (`vc_grow_seg_from_seed.cpp:396-505, 1466-1520`): seed from
  an existing segment's edge, reject when ≥N segments already cover the point,
  write the symmetric overlap graph. This is the natural feeder for our fusion path.
- **`gen_neighbor`** (`:614-880`): derive the *whole* next wrap by normal-casting
  every vertex with a clearance/exit state machine, then fold-repair by binary
  search and interpolate misses. Two ideas port immediately into our single-ray
  seeder at `:3864-3874`: require the ray to *exit* the current sheet before
  accepting a hit, and cast along the local surface normal rather than the
  umbilicus radial.
- **Periodic `optimize_surface_mapping`** (`GrowSurface.cpp:989-1385`): hole
  closing, re-flatten, re-derive donor membership, delete unsupported cells, with
  a rollback if the valid count halves. Our periodic solves relax geometry but
  never re-derive membership and never delete a cell.
- **Best-snapshot rollback** (`:1888-1897, 3505-3520`) around any pass that can
  shrink the surface — three memcpys per generation, negligible next to solve cost.
- **Lookahead/rollout growth** (`:2439-2726`) — real but expensive; the cheap half
  (the L-shape rule) is folded into G6.
- **Approved/defective donor tiering** (`:1462-1470, 2849-2876`) — matters as soon
  as the donor set is a library rather than a hand-picked pair, and is *required*
  once G6's consensus gate exists, or a lone approved donor can never reach threshold.
- **Cross-wrap row alignment / `vc_straighten`** (`vc_straighten.cpp:480-645`):
  pair each cell with its counterpart one winding away and remove the systematic
  `dv` drift. Measure first — `mean|dv|`, `p95|dv|` — it is nearly free once
  G4's ray machinery exists, and we currently do not even know how large our
  drift is.
- **Lateral (fractional) registration between wraps** (`spiral_ceres.hpp:180-207`):
  in ribbon multi-wrap mode nothing ties column i of block b to column i of block
  b+1 — the self-overlap hinge is purely radial and indifferent to tangential
  slip, so blocks drift angularly and the saved u parameterisation is sheared,
  which is exactly what `vc_cut_windings` assumes is coherent. ~20 lines reusing
  the segment projection in `tr_res_space`.
- **Coarse-to-fine `growth_scale`** (`GrowPatch.cpp:3120-3152`) — a cheap-and-wide
  mode we lack entirely; blocked on `ng_open` reading only one level.

## 3. Divergences that are deliberate, already handled, or overstated

- **`TRF_SPACE` disabled when normal grids are active** (`:3024-3025`) — matches
  vc3d, which ships the space-line term off. Correct as-is; it only raises the
  cost of G14.
- **Commit-before-solve** — vc3d does this too; the gap in G1 is the *missing
  pre-veto and the missing retraction*, not the ordering.
- **Global solve every 8 generations forever** (vc3d: first 10 only) — a
  deliberate, documented divergence justified by our cheaper parallel sweeps.
  Note it is partly compensating for the missing candidate ordering (G6.1); keep
  it, but expect to be able to reduce it once ordering lands.
- **Anchor accuracy is *not* unreported.** `:4180-4192` prints a per-anchor final
  distance and an explicit "never captured" line. The residual gap is narrower
  than claimed: no anchor metrics in meta.json, and no same-winding
  distance-ratio consistency test between anchor pairs (`surface_metrics.cpp:118-186`).
- **The save-time fill does not fabricate surface beyond the traced region.** It
  only touches cells with `state == R3D_TR_SET`, i.e. cells the tracer actually
  grew; it never invents cells outside the grown area. The real defect (G8) is
  narrower but sharper: it re-seats *tear-cut* cells, undoing the wrong-wrap
  safety check applied a few lines above, and it is data-blind.
- **`dsup` is not unused.** It is a conf floor and a wrong-wrap veto in
  `tr_spiral_flag`. G6 is specifically that it does not gate *placement*.
- **Umbilicus winding prior, winding-potential field, self-overlap hinge, ribbon
  multi-wrap seeding, the 4-wide fused `ncp_residual4`** — capabilities vc3d does
  not have. Keep. G2/G3 fix the fused path's gradient, they do not argue against
  the fusion itself.
- **`TR_DT_TH` = 128 vs vc3d's 170** (`GrowPatch.cpp:387, 3078-3079`) is a real
  divergence but *not* obviously a bug: at 128 the zero-distance set includes the
  soft fringe of each prediction blob, so the DT plateau is ~2× thicker, the
  space-line term loses its gradient across the fringe, conf saturates off-centre,
  and the crossing detector (`prev < 2.5`, `:3873`) can merge sheets whose fringes
  touch. If 128 was tuned against our prediction volumes, document it as such and
  expose it as `cfg.pred_th`; if it was inherited, it is wrong. Re-check the three
  constants calibrated against it (seed probe 64.0, `prev < 2.5`, ribbon stop 50.0)
  either way.
- **`NG_QMAX` 64 / `NG_HOODSEG` 384 truncation** has no vc3d counterpart and cuts
  in bucket/path order rather than by distance — real, but instrument before
  spending: add a counter next to `tr_tm_ns` to confirm the cap ever binds on real
  xz/yz slices, and if it does, make the cut distance-ranked and raise to ~1024.
- **Coarse-to-fine growth scale and rollout lookahead** — genuinely absent, but
  both are throughput/robustness features whose value depends on run sizes we are
  not yet hitting. Deprioritised, not dismissed.

## 4. Recommended implementation order (top 5)

1. **G1 — pre-solve candidate veto + xyz box + disc restore on failure.**
   ~80 lines, no new structures, removes a systematic bias toward dead regions on
   every clamped or ribbon run, and pays for itself in wall clock. Do the ribbon
   CT test move in the same change.
2. **G2 + G3 — freeze the normal-grid configuration across the FD probes, and
   return exactly 0 with no coverage.** Same function, one sitting; affects every
   solve on every grid-active run and removes a *wrap-jump force* plus a global
   quad-flattening torque. Highest quality-per-line in the list.
3. **G7 seed snap + G6.1 candidate ordering.** Two small, independent,
   low-risk changes with immediate measurable effect: the seed one determines the
   sheet choice for the entire trace; the ordering one removes the cause of the
   nondeterminism the periodic global solve is currently paying to hide, and is
   the prerequisite for G6's acceptance gate.
4. **G4 — winding relaxation + `werr` confidence clamp + geometric wrap QC.**
   The first change that makes wrong-wrap capture *visible* rather than silent,
   and it un-blinds the five existing defences that read a stale winding value.
   Land the measurement half first; the conf clamp is a two-line follow-up.
5. **G11 — `gen_of` stamp + `generations.tif` + rewind + tifxyz load.**
   Not a quality improvement by itself, but it is the enabling substrate for
   every correction workflow (G10's region regrow, resume, expansion), and it is
   the difference between "re-trace from scratch" and "fix the bad 30 generations".

Immediately after: **G5** (gap field + signed spacing) and **G8/G9** (gated,
solve-based inpaint with the staged schedule) — G9's weight-scaling plumbing is
shared, so do it once and let both consume it. **G14** before the next published
normal-grid store, whichever release that is.
