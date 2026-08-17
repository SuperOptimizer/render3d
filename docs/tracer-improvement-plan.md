# Tracer improvement plan

Derived from `vc3d-gap-review.md` (16 confirmed gaps, G1-G16). This is the
execution order, with dependencies, effort, risk, and — because "improved"
must be measurable — the verification harness each phase reports against.

Guiding principle from the review: our inner loop is faithful; what's missing
is the outer machinery — *veto before commit, undo after commit, measure the
result*. The plan front-loads (a) measurement, so every later phase has a
number, and (b) the highest quality-per-line force fixes.

## Verification harness (used by every phase)

- Reference runs: `R3D_TRACE_TEST` at 3 pinned seeds on PHerc0343 L1
  (60 gens) + 1 ribbon run, same seeds forever. Script:
  `tools/trace_bench.sh` (to be written in Phase 0) writes a JSON row per
  run: nset, area_cm2, fill fraction, hole fraction, folds/kinks/twist,
  wrap-jump fraction (once G4 lands), donor rms (fused run), wall time.
- A/B rule: every phase lands only with a before/after table appended to
  `docs/measured.md`. A change that moves QC numbers the wrong way reverts
  or gets a flag.
- Determinism note: until Phase 2's candidate ordering, same-build runs vary
  (documented 0.06%..9.7% v-edge swing); compare medians of 3 runs.

## Phase 0 — measure first (G15 core, instrumentation)   [~1 day]

Everything else is judged by these numbers, so they land first.

1. `tr_qc` extensions: per-quad two-triangle **area** (vx² and cm² via um),
   valid **bbox** + fill fraction, **enclosed-hole fraction** (border
   flood-fill — same machinery G8 needs later), **slant** tau =
   (e_u·e_v)/|e_u|² (the only detector for coherent shear).
2. `tr_qc_donor` (fused runs): sample ≤2000 kept cells → nearest donor
   distance mean/rms/p95 + coverage both directions.
3. Emit all of it into meta.json + the finish log + the tracer panel.
4. `NG_QMAX`/`NG_HOODSEG` bind counters next to `tr_tm_ns` (decides whether
   the truncation gap is real before spending on it).
5. `tools/trace_bench.sh` + baseline rows for the pinned seeds.
- Risk: none (read-only passes). Unblocks: acceptance criteria for all phases.

## Phase 1 — fix the forces (G1, G2, G3)   [~2 days]

The physics must be right before tuning anything above it.

1. **G1 pre-solve veto + xyz box + disc restore.**
   - `x_min/x_max/y_min/y_max` in cfg; valid-3×3-mean test in
     `tr_place_cand` *before* touching pos/state.
   - Move the ribbon CT/DT test into the same pre-veto (currently a full
     generation late).
   - Snapshot the radius-3 disc before the placement solve; memcpy back on
     post-solve failure (keeps the second line of defence scar-free).
   - Verify: clamped + ribbon reference runs — expect fewer FAIL scars along
     the clamp, wall-clock drop (no more 50-iter solves on doomed cells).
2. **G2 freeze the normal-grid gradient configuration.** Hoist `aw`, freeze
   straddle corner + slice, compute ROI accept/1/d² weights from variant 0,
   snap-search once and re-evaluate frozen endpoints for probes. Mirror in
   scalar `ncp_residual`.
3. **G3 zero on no-coverage.** Return 0 when the hood has no segments; the
   1.0 no-target penalty only when ROI-accepted segments exist.
   - Verify (G2+G3 together): all reference runs — this removes a documented
     wrap-jump force and a global quad-flattening torque; expect folds/kinks
     and wrap behaviour to improve everywhere grids are active. Also compare
     per-cell solve cost trace (Jacobian discontinuities gone).
- Risk: medium (heart of the data term) — guard with `R3D_NCP_LEGACY=1` env
  fallback for one release; A/B both ways.

## Phase 2 — right start, right order (G7, G6.1, G6.3)   [~1 day]

1. **G7 seed snap**: `tr_seed_snap` before the seed quad — walk the radial
   (or ±axes) for the nearest DT minimum < 2.5; distinct error code so the
   GUI says "seed is not on a sheet"; snapped seed shown in the panel.
2. **G6.1 candidate ordering**: sort `cands[]` by (support run-depth desc,
   SET-neighbour count desc, index) — certain territory first; removes the
   nondeterminism the every-8th-gen global solve pays to anneal away.
3. **G6.3 L-shape rule**: reject candidates that close no 2×2 block after
   generation ~30 — kills 1-cell spurs the anti-fold hinge currently fights.
- Verify: variance across same-build triples should collapse (target: the
  QC swing well under the documented 9.7% worst case); seed-snap A/B on
  deliberately-offset seeds.
- Later dividend: consider relaxing the every-8th-gen global solve cadence
  once variance is gone (wall-clock win, measured).

## Phase 3 — winding integrity (G4, then G5)   [~3 days]

1. **G4a wind relax**: `tr_wind_relax` in the `tr_spiral_fit` slot — Jacobi
   over SET cells with the seed pinned, edges rejected at |Δθ/2π| > 0.25.
2. **G4b werr conf clamp**: conf ≤ 0.25 where |w − relaxed| > 0.3 — makes
   wrong-wrap capture *visible* and un-blinds the five defences that read
   stale winding (hinge, spacing, donor gate, adoption gate, sibling stop).
3. **G4c `tr_qc_wrap`**: geometric second opinion via the existing `tr_sfx`
   hash — normal-ray self-crossings → wrap-jump fraction + wrap_err p95 into
   QC. Land the measurement before the clamp; watch it for a week of runs.
4. **G5a gap field**: `sp_om_field` (32-voxel cells, nearest-fill + box
   smooth, global median fallback) behind `tr_om_at(t, p)`; every
   `tr_om_eff` call site routed through it.
5. **G5b normal-ray gap measurement** (radial ray overestimates by 1/cos on
   pancaked sections).
6. **G5c signed spacing**: sign from grad(r1)/umbilicus radial ×
   sign(Δwinding); inverted branch escapes the Cauchy robustifier so the
   repair force survives.
- Verify: wrap-jump fraction (new in G4c) on all reference runs; the
  self-overlap hinge's behaviour on the multi-wrap ribbon run; gap-field
  visualization dump for one slice.
- Risk: G5c changes an existing residual's shape — flag-guard
  (`R3D_SIGNED_SPACING=0` fallback) for one release.

## Phase 4 — corrections that actually correct (G9, G10, G11, G13)   [~4 days]

The anchor/re-solve workflow exists; this phase makes it converge to the
*right* answer instead of the nearest basin.

1. **G9 staged weight relaxation** (shared plumbing first): `wdist/
   wstraight/wsnap` in the solve context; refine passes at {0.3, 0.1, 0} →
   {0.3, 0.1, 1} → {1, 1, 1}; radius widened to 8 + anchor spread. Inpaint
   (Phase 6) reuses the same schedule.
2. **G10a tangential-only reopt anchors**: remember position at refine
   start, penalise only tangential drift (w=10), softplus cap on normal
   motion at the disc edge — `TRF_REOPT`, refine/inpaint only. Stops the
   radius-8 anneal from silently shearing uv against the untouched region.
3. **G10b quad-plane correction loss**: replace the 3 per-axis anchor
   residuals with vc3d's plane distance over 4 quads with 40-voxel/2-cell
   falloffs; the sheet slides tangentially at zero cost (a correction is
   about *which sheet*, not which uv). Drop TR_W_ANC to 1.0.
4. **G11 generation stamp**: `gen_of` per cell, `generations.tif` channel,
   `r3d_tracer_load` (tifxyz → tracer), `r3d_tracer_rewind(gen)`; panel
   gets a rewind slider. This is the substrate for "fix the bad 30
   generations" instead of re-tracing.
5. **G13 growth steering**: `grow_dirs` bitmask (default {±u} in ribbon
   mode — stops the ±v/diagonal leak the z-anchor fights), optional
   `grow_mask`.
6. **G10c region regrow** (after G11+G13): flood-fill low-conf region from
   an anchor, freeze the boundary ring, EMPTY the interior, regrow inside
   the mask. Where the patch jumped a wrap, this is the only clean fix.
- Verify: scripted anchor-correction scenario — trace with a known
  wrong-wrap capture (reproduce via seed choice), drop anchor on the right
  sheet, re-solve; assert final anchor distance ~0 *and* wrap-jump fraction
  drops *and* no uv shear (slant metric stable). The existing
  R3D_ANCHOR_TEST/R3D_REFINE_TEST hooks extend naturally.

## Phase 5 — fusion worth the name (G6, G12, donor tiering)   [~4 days]

1. **G6 consensus gate + annealed retry** (needs G6.1 from Phase 2):
   `tr_eval_cell` (0-iteration cost probe); inlier vote ≥2 donors within
   tolerance; rejects go back to EMPTY (retryable), not PROC; `inl_th`
   anneals 20→10 with fringe reseed at each generation-empty, emergency 2.
   Guard on `t->don` so raw tracing keeps GrowPatch semantics.
2. **G12a donor uv membership**: ≤4 (donor id, uv) slots per cell; affine uv
   extrapolation as the candidate initial guess (trim-worst); one pull per
   slot toward `donor_bilerp`; wrong-fold rejection = extrapolated-vs-
   nearest uv disagreement > 2 cells.
3. **G12b high-res resample at save**: emit through donor parameterisation
   at 1/src_step with outlier rejection + multi-donor averaging — inherits
   the donors' fine geometry instead of the coarse quad grid.
4. **Donor tiering** (approved/defective, from G16) — required once the
   consensus gate exists or a lone approved donor can never reach threshold.
- Verify: fused reference run — donor rms/p95 (Phase 0 metric), coverage
  both directions, hr output diffed against donors in their own uv.

## Phase 6 — save, inpaint, pipeline (G8, G15 rest, G14, G16)   [~4 days]

1. **G8 gated, solve-based inpaint**: border flood-fill interiority gate
   (shares Phase 0 machinery); membrane as initial guess only; then
   `tr_local_opt` over hole cells with the G9 schedule (geometry → data-term
   ramp → TRF_ALL); conf recomputed so evidence-free fills stay below the
   save cutoff; tear-cut cells never re-seated (the current fill undoes the
   tear mask — that specific bug can be cherry-picked earlier if needed).
2. **G15 rest**: min-area gate (refuse save + distinct error), bbox crop
   with `grid_offset` in meta, anchor metrics into meta.
3. **G14 normal-grid store metadata**: `sparse-volume` snap + strided
   prefetch; multiscale format (level dirs, `coordinate_scale`, per-level
   metadata); store spiral-step becomes authoritative for `cfg.step`.
   *Deadline-driven: must land before the next published store.*
4. **G16 orchestration** (in descending value):
   - single-ray seeder upgrades: require sheet *exit* before next hit; cast
     along local normal (2 small fixes, can land any time);
   - ray-cast multi-seed generator (one seed per crossing, snapped to
     raw-CT mid-papyrus);
   - expansion mode (seed from segment edges, overlap-graph output) — the
     feeder for fusion;
   - best-snapshot rollback around shrink-capable passes (3 memcpys/gen);
   - periodic membership re-derivation + unsupported-cell delete with
     halve-rollback;
   - lateral inter-wrap registration for multi-wrap ribbons (~20 lines
     reusing `tr_res_space` projection);
   - cross-wrap dv drift: *measure* (mean/p95 |dv|) once G4 rays exist,
     decide on `vc_straighten` port from data.

## Explicitly deferred (measured before built)

- `TR_DT_TH` 128 vs 170: expose as `cfg.pred_th`, run the A/B against our
  prediction volumes, re-check the three constants calibrated against it
  (seed probe 64, crossing 2.5, ribbon stop 50). Document the winner.
- `NG_QMAX`/`NG_HOODSEG` truncation: act only if Phase 0's counters show the
  cap binds on real slices; then distance-ranked cut at ~1024.
- Coarse-to-fine growth scale + rollout lookahead: revisit when run sizes
  hit the wall; growth-scale is blocked on G14's multiscale support anyway.

## Effort summary

| Phase | Content | Est. | Quality leverage |
|---|---|---|---|
| 0 | metrics + harness | 1d | enables everything |
| 1 | G1, G2, G3 | 2d | highest per line |
| 2 | G7, ordering, L-shape | 1d | determinism + right start |
| 3 | G4, G5 | 3d | wrong-wrap made visible, then rare |
| 4 | G9, G10, G11, G13 | 4d | corrections that converge right |
| 5 | G6, G12, tiering | 4d | fusion quality |
| 6 | G8, G15, G14, G16 | 4d | save/pipeline integrity |

Order within a phase is a dependency chain; phases 2 and 3 can overlap with
phase 1's soak. Every phase ends with the reference-run table in
`docs/measured.md` and a commit per gap (revertable independently).
