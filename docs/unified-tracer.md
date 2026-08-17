# The unified tracer

Goal: one tracer where grow-from-seed, ribbon/Lasagna cross-sections,
multi-wrap growth, the spiral/winding machinery, neighbor-wrap derivation,
per-slice global arbitration, fusion, and corrections are **operators over
one shared state**, not separate modes — so a whole scroll comes from one
loop instead of one patch per click.

## The unifying observation

Fusion is already the universal glue. Anything that produces a surface —
a grown patch, a loaded segment, a *derived* neighbor wrap, a corrected
region — can be a donor, and the fused grower (consensus gate, donor uv,
affine guesses, fold veto) turns donors + gaps into a clean new surface.
So the unified tracer is not a rewrite: it is a **driver loop** over the
operators we already have, plus two missing operators (derive, arbitrate).

## Shared state (exists today in r3d_tracer, minus the registry)

- the quad grid(s) with per-cell winding, conf, werr, generation
- the winding frame: umbilicus + spiral fit + winding-potential field
- the per-region sheet-gap field (tr_om_at)
- the donor set with uv membership
- QC maps (folds/kinks/twist/wrap/donor agreement)
- NEW: a **wrap registry** — the set of saved wraps with their winding
  offsets (spiral.json already carries the registration)

## Operators

| operator | status | role in the loop |
|---|---|---|
| seed (snap, multi-crossing) | done | entry point |
| grow (GrowPatch inner loop) | done | raw growth where no donor exists |
| fuse (consensus + uv + affine) | done | growth guided by donors |
| correct (anchors, rewind, reopt) | done | operator-in-the-loop repair |
| measure (QC) + decide (anneal/retry) | done | acceptance |
| **derive** (gen_neighbor) | **stage 1** | wrap N -> initial wrap N±1 |
| **arbitrate** (Lasagna maxflow) | stage 3 | global per-slice wrap assignment |
| gap solve (spiral service) | stage 4 | solved per-point gaps replace the field |

## The loop (stage 2 driver)

```
wrap[0] = grow(seed)                     # one good wrap, human-checked
repeat outward and inward:
    guess = derive(wrap[k])              # normal-cast one gap over
    wrap[k+1] = fuse_grow(guess ∪ neighbors)  # consensus keeps it honest
    register(wrap[k+1])                  # winding offset via spiral frame
    if QC bad: arbitrate slice-wise / flag for correction; stop this arm
save each wrap; the registry IS the unrolled scroll
```

Ribbon mode folds in naturally: a ribbon is `grow` with a z-slab policy
and per-slice rows; multi-wrap ribbons are the same loop running its arms
in lockstep. Lasagna's per-slice view enters twice: ribbons as a growth
policy, maxflow as the arbitration operator.

## Stage 1 — derive: `tr_derive_wrap` (gen_neighbor port)

From a finished grid, cast every trusted cell along its local normal by
the LOCAL gap (tr_om_at — this is why the gap field exists), with vc3d's
clearance/exit state machine: the ray must first EXIT the current sheet
(DT rises above a clearance), then the first DT minimum is the neighbor
crossing. Misses interpolate from hit neighbors; fold-overlaps repair by
binary search between adjacent hits. Output: a tifxyz written next to the
source (`<dir>-next`/`-prev`), winding shifted ±1 — immediately a donor
for `fuse_grow`, no new infrastructure.

CLI: `tracecli <pred> --load DIR --derive +1 --out DIR2`
GUI: "derive next wrap" button on a done trace.

## Stage 2 — the driver

`tracecli --unroll N`: run the loop N wraps each way from the loaded/
traced wrap; per-wrap QC rows; stop an arm when donor coverage or wrap QC
collapses. GUI mirrors it with a wrap list in the tracer panel.

## Stage 3 — arbitrate (Lasagna maxflow)

Port `vc_lasagna_maxflow_graph`: per cross-section slice, nodes = radial
sheet crossings, edges = continuity along the slice + gap consistency;
max-flow assigns crossings to wraps globally. Runs at arm boundaries to
settle regions where two wraps claim the same crossing — the wrong-wrap
failure mode, solved globally instead of by local hinges.

## Stage 4 — solved gaps

Replace the coarse gap field with the spiral service's model: per-point
signed distances to both neighbor wraps as unknowns, smoothness +
minimum-separation + radial-order constraints (the full G5). The derive
operator and the self-overlap hinge both read it.

## Non-goals

- No new file formats: wraps are tifxyz + winding + spiral.json.
- No monolithic rewrite: each stage lands as an operator + driver change,
  benchmarked (rendered images + QC rows) before the next.
