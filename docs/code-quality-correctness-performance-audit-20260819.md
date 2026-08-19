# Render3D code quality, correctness, and performance audit — second pass

**Date:** 2026-08-19

**Audited revision:** `1c14a98dbfbab54f6607489f85b0375de0f01430`, including the uncommitted worktree present on the audit date

**Relationship to the first audit:** this is a separate adversarial review of correctness, data integrity, scale, memory use, persistence, APIs, build quality, and testing. Thread ownership, GUI blocking, cancellation, and concurrent publication are covered in `docs/threading-nonblocking-audit-20260819.md` and are not repeated here except where they directly change correctness.

**Method:** three focused agents independently reviewed GUI/render/Vulkan, streaming/I/O/scale, and processing/services/build surfaces. Their findings were then cross-checked against the live tree and ranked by observed failure mode rather than code size or stylistic preference. Release, ASan/UBSan, TSan, shader-build, and CTest configurations were exercised. No product code was changed.

The worktree was already dirty before this pass. Tracked changes existed in `CMakeLists.txt`, `src/core/input.[ch]`, `src/core/tracer.[ch]`, `src/main.c`, `src/render/render.h`, `src/shaders/raycast.slang`, `src/shaders/shaders.cmake`, `src/vk/vkbackend.c`, and `tests/test_quick.c`; new source files included `src/core/{bsurf,labelvol,regvol}.[ch]` and `src/shaders/postfilt.comp`. This report describes that actual state.

## Executive verdict

Render3D contains good low-level work: the renderer manifest parser has meaningful limits, the shard reader performs strong bounds and integrity validation, several queues are bounded, CMake exposes useful warning/sanitizer configurations, and the existing geometric and segment tests cover important happy paths. Those strengths should be retained.

The code is nevertheless **not yet safe to describe as a correctness-hardened massive-volume tool**. The second pass found release-blocking ways to:

1. turn malformed remote responses into durable zero-filled or partial cache data;
2. lose the only trace/label result after a failed save;
3. resume a saved trace with altered configuration and no remaining growth budget;
4. resurrect erased labels or erase valid in-memory labels while loading a corrupt brick;
5. permanently cache a false-empty predictor result;
6. publish a partial segment corpus after allocation failures or reuse stale packed grids;
7. execute shell syntax derived from remote object names;
8. heap-overflow renderer page-table initialization with a syntactically accepted manifest;
9. write beyond a 2x2 offscreen storage-image region after an incomplete viewport clamp;
10. exhaust tens of gigabytes through legal chunk dimensions, dense metadata, whole-plane caches, duplicated decoded caches, and converter worker fan-out.

The main architectural problem is that limits and failure semantics are local rather than systemic. A subsystem may bound a queue but not the bytes behind it; validate a manifest in the Vulkan path but parse the same schema less safely in the CPU path; atomically rename one file while publishing a multi-file artifact nontransactionally; or report success after an allocation/write failure changed the output. For a tool operating on massive and sometimes remote data, every externally derived size, identity, and artifact generation needs one checked, versioned, fail-closed contract.

## Severity and release standard

- **P0 — release blocker:** silent data corruption/loss, code execution from external names, unsafe parsing that can produce invalid memory access, or a fundamental scale path that predictably reaches multi-gigabyte amplification without a budget.
- **P1 — required hardening:** substantial correctness ambiguity, nontransactional persistence, routine multi-gigabyte memory use, silent truncation, weak lifecycle/API ownership, or a major test gap around these paths.
- **P2 — maintainability/performance debt:** superlinear algorithms, portability and shader-contract drift, unsafe public preconditions, duplicated parsers, or missing diagnostics that materially raise future defect risk.

A release gate should require:

- malformed, truncated, oversized, incompatible, or partially written input never becoming a valid cache/artifact;
- the last known-good artifact remaining readable after every simulated allocation, write, flush, rename, network, and process-crash failure;
- every large allocation and queue charged to an explicit process-wide byte budget;
- checked addition/multiplication before converting dimensions/counts to `size_t`, `uint32_t`, protocol fields, or GPU descriptors;
- sanitizer and corruption/fault-injection suites as required CI jobs, not optional local presets;
- documented format versions, endianness, identity/provenance, length limits, and checksums.

## What was actually run

| Check | Result | Interpretation |
|---|---|---|
| Release build | Current; `ninja: no work to do` | The existing release tree was buildable. |
| Release quick tests | 2/2 passed in about 0.11 s | Useful smoke coverage, but not evidence for failure paths or concurrency. |
| Dev ASan+UBSan configure/build | Passed | The current sources compile in the sanitizer configuration. |
| Dev sanitizer quick tests | **Failed** | LeakSanitizer reports a direct 6,144-byte leak from `r3d_tracer_spiral_selftest` (`src/core/tracer.c:6652`); cleanup at `6715-6723` omits `t.werr`. |
| Full dev CTest | 1 failure, 3 passes, 1 data-dependent skip | Quick failed on the leak; shard safety, C5D GPU, and GPU tests passed; the optional shard-data test was skipped. |
| TSan quick label | 2/2 passed | The existing quick tests do not exercise the cache/lifecycle/worker interactions identified by the threading audit, so this is a narrow result. |
| Shader compilation | Completed with warnings on every raycast variant | Slang implicitly upgrades the requested `spirv_1_5` profile because generated shaders require additional capabilities. |
| `spirv-val --target-env vulkan1.2` on generated variants | Passed | No validation failure was reproduced, but capability drift is not currently asserted by the build. |

## P0 findings — correctness, integrity, and security

### P0.1 — Malformed HTTP 200 responses can permanently poison the brick cache

Remote ingest parses Blosc metadata without first proving that the response contains a complete minimum header (`src/vk/vkbackend.c:2749-2758`; CPU analogue `src/core/cpuvol.c:345-355`). A malformed chunk is skipped rather than marking the cell transaction failed. If no chunk succeeded, the renderer writes definitive empty markers; if some chunks succeeded, it encodes bricks with the missing region left as zero (`src/vk/vkbackend.c:2779-2839`; `src/core/cpuvol.c:365-415`). The response writers grow without a maximum or checked `size_t` growth (`src/vk/vkbackend.c:2533-2546`; `src/core/cpuvol.c:181-196`).

An HTML proxy page, short body, corrupted frame, incorrect declared sizes, or oversized response can therefore cause invalid header access, memory exhaustion, or durable false air/partial CT data. Subsequent healthy requests may never repair it because the poisoned artifact now appears cached.

**Required change:** define one remote-chunk validator used by all consumers. Check minimum header length before calling Blosc, checked declared compressed/uncompressed sizes, exact expected voxel bytes, HTTP status/content constraints, configured response cap, and source checksum/version where available. Any malformed expected chunk fails the entire cell transaction; it must create no `.c5b` or negative marker. Quarantine diagnostics separately. Negative caching for 404 requires an immutable source identity or a bounded TTL.

**Acceptance:** inject zero-byte, 8-byte, HTML, bad `cbytes`/`nbytes`, bit-flipped, truncated, and oversized 200 responses. No cache artifact is published, ASan/UBSan remain clean, peak body memory stays under the configured cap, and a later healthy retry renders the exact source bytes.

### P0.2 — Tracer save/load is not a faithful round trip and can make resume a no-op

Save crops to a kept bounding box and writes `grid_offset` and `max_gen` (`src/core/tracer.c:6483-6525,6557-6579`). Load reads only scale, derives `cfg.max_ring` from the cropped width, hard-codes threshold/level, reconstructs generation from the maximum stored value, loses anchor coordinates, and resets confidence (`src/core/tracer.c:5847-5926`). Grow then computes its budget from the inferred `max_ring - gens_done` (`src/core/tracer.c:5277-5284,6191-6196`). A cropped saved result can reload with zero remaining budget even though the user expects rewind/refine/grow behavior promised by `src/core/tracer.h:233-267`.

This is semantic corruption rather than harmless metadata loss: the same artifact no longer represents the same solver state and subsequent computation can differ or do nothing.

**Required change:** use a versioned manifest containing the original coordinate frame/origin/dimensions, complete solver configuration, maximum generation, anchors, confidence/QC fields, source identity, and artifact checksums. If a legacy file lacks sufficient state, label it explicitly as a display/import surface rather than silently treating it as resumable solver state.

**Acceptance:** save -> destroy -> load -> snapshot and save -> load -> grow equivalence tests for cropped, non-square, anchored, rewound, and partially grown traces. Compare configuration, coordinates, confidence, generations, and deterministic next-step output.

### P0.3 — Persistence failure can destroy the only in-memory result

Auto-harvest attempts directory creation and tracer save, then unconditionally frees trace/display buffers and marks the trace inactive (`src/main.c:5194-5220`). Disk full, permission failure, TIFF failure, or a save cutoff can therefore discard the only result. Dataset-swap/shutdown label autosave similarly ignores the return value before freeing (`src/main.c:6457-6462`), and final umbilicus persistence is also ignored (`src/main.c:6397-6399`).

**Required change:** make artifact lifecycle explicit: `READY -> SAVING -> PERSISTED` or `SAVE_FAILED_RETAINED`. Persistence operates on an immutable snapshot; live state remains until a complete generation is durably published or the user explicitly chooses to discard it. Return structured errors with the failed artifact/path and retryability.

**Acceptance:** inject `ENOSPC`, `EACCES`, short writes, TIFF errors, and failed rename/fsync at every stage. The in-memory result stays usable, the prior artifact remains intact, and the UI exposes retry/export-as/discard rather than silently continuing.

### P0.4 — Label persistence can resurrect erased labels or erase valid live labels

The save path writes `manifest.json` directly before its brick transaction and ignores `fprintf`/`fclose` errors (`src/core/labelvol.c:208-225`). When an empty brick should remove old content, `unlink` failure is ignored but the live generation is marked saved (`241-244`), so stale labels can return on the next load. Temp names are fixed rather than uniquely created (`254-260`).

Load ignores `manifest.json`, so it does not validate format version, brick edge, dataset dimensions, or identity (`273-326`). It subtracts the old brick's counts and decodes directly into live storage. On decode failure it zeros that live brick, updates counts/generation, and marks it saved anyway (`300-318`). A valid empty label set is reported as failure because success is `loaded > 0` (`323-326`). File length comes from unchecked `fseek`/`ftell` and has no encoded-size cap (`289-292`).

**Required change:** validate a versioned dataset-bound manifest first; decode every brick into bounded temporary storage; validate class values and checksum; then atomically swap. Save dirty immutable snapshots to unique temporary files, fail if removal fails, publish the manifest last, and mark clean only if the current live generation still equals the saved generation. Add directory fsync where crash durability is claimed.

**Acceptance:** empty round trip, wrong-dataset manifest, corrupt/truncated/oversized brick, class corruption, unlink failure, concurrent generation change, disk full, and crash at every publish phase. A failed load preserves the entire prior live label volume, and a failed save remains dirty.

### P0.5 — Predictor emptiness sampling can permanently cache a false-empty result

The predictor checks only every 4,096th byte to decide whether an input brick contains signal (`src/core/surfpred.c:258-273`). A brick whose nonzero voxels do not fall on those offsets is classified empty. A zero prediction is then thresholded and persisted (`274-320`). Completion is inferred from file existence (`129-141`), while the writer ignores directory, encode, and write failures and has no error return (`160-185`).

The result can be both false and sticky: future runs see a file and skip the actual inference.

**Required change:** use a complete vectorized/reduced nonzero test or reliable source metadata; make persistence return errors; publish versioned artifacts with source/model/config provenance, exact dimensions, checksum, and completion marker only after validation.

**Acceptance:** nonzero voxels at offsets 1, 4095, 4097, and the last byte must all trigger inference. Corrupt, partial, wrong-model, and failed-write outputs must not count as complete.

### P0.6 — Segment-store rebuild can publish a partial or stale corpus

Carry-forward allocation failures only break their loops (`src/core/segstore.c:273-277,319-325`); build then writes whatever subset remains as the new manifest (`337-354`). A transient OOM can silently remove existing segments from the active corpus. Basenames are truncated to 63 characters and used as identity/path without collision checks (`19-29,210-245`). Existing `.tfx` files are reused merely because they exist (`103-110`), so changed source data with the same name can be paired with stale packed geometry. Errors during decimated backfill are ignored (`292-303`).

The binary manifest is a dump of native structs with implicit padding and endianness (`13-17,337-350`). Open uses externally supplied counts in unchecked size arithmetic before allocation/copy (`357-386`) and lacks a version, endian marker, checksum, and strict count caps. `tile_ofs + tw*th` is not comprehensively overflow-checked.

**Required change:** abort the entire rebuild on any allocation/encode/write failure and retain the old corpus. Give sources collision-resistant IDs and fingerprints, validate reused grids against dimensions/hash/codec, and serialize explicit little-endian fields with version, checked lengths/offsets, and checksums. Publish a complete manifest generation atomically.

**Acceptance:** allocator failpoints at every growth step preserve the old corpus; same basenames in different directories, names longer than 63 bytes, modified sources, corrupt counts/offsets, truncation, cross-endian fixtures, and crash points all fail closed.

### P0.7 — Remote object names cross a shell-command boundary

Open-data names derived from S3 listings are interpolated into single-quoted shell command strings (`src/main.c:1340-1367,1402-1419,1443-1454,1475-1484`) and executed through `popen` (`1055-1064`). A key containing an apostrophe can terminate the quote and inject shell syntax. Traversal-like names also complicate containment even without shell metacharacters.

**Required change:** remove the shell boundary. Use direct library APIs or `posix_spawn`/`fork` with an argv array, explicit environment, owned PID/process group, and validated path components constrained beneath the intended root. Do not attempt to solve this with another quoting helper.

**Acceptance:** apostrophes, whitespace, newlines, `$()`, backticks, semicolons, leading dashes, `..`, absolute paths, and Unicode edge cases are treated solely as data; no command substitution occurs and no output escapes the selected root.

### P0.8 — Legal source chunk sizes cause catastrophic download and memory amplification

Remote ingest defines its work cell as `LCM(chunk_edge, 128)` (`src/vk/vkbackend.c:2478-2490`). Each worker retains a full source chunk plus full LCM cell and response body (`2649-2652`), accepts chunk edges through 1,024 (`3760-3764`), and defaults to as many as 16 workers (`3777-3788`). The CPU-volume path repeats the cell algorithm (`src/core/cpuvol.c:199-206,312-417`).

Representative consequences:

- edge 32: 64 HTTP requests to assemble one 128-cubed brick;
- edge 192: 8 requests and 27 output bricks transcoded for one cell;
- edge 1,024: one visible-brick request creates a 1 GiB chunk/cell, transcodes 512 bricks, and raw response + chunk + cell storage approaches 3 GiB per worker — theoretically about 48 GiB across 16 workers.

This is a supported-input denial of service and violates the massive-volume premise even before GPU or decoded caches are counted.

**Required change:** cache source chunks directly and assemble only overlaps required by the requested brick. Neighbor transcodes are optional work behind a global byte/CPU budget. Cap response, source chunk, per-request, and total in-flight bytes; derive concurrency from that budget, not CPU count alone.

**Acceptance:** for source edges 32, 64, 192, 512, and 1,024, first-visible-brick traffic is no more than twice the mathematically overlapping bytes, no single allocation exceeds the chosen limit (for example 256 MiB), and process RSS stays within the configured cache plus in-flight budget.

### P0.9 — Renderer metadata is dense enough to consume gigabytes before voxel data

The LOD renderer allocates multiple arrays per virtual brick: page shadow, slot/want/max fields, a 12-byte candidate, and three warm-state arrays (`src/vk/vkbackend.c:304,3917-3960`), roughly 38 CPU bytes per brick plus about 4 bytes per mapped page entry (`493-518`) and optional remote/overlay bytes (`3770-3771,4311-4316`).

For the repository's 43,008 x 43,008 x 68,608 geometry (`src/vk/vkclip.c:283-304`), an eight-level 128-cubed pyramid contains about 69.2 million virtual bricks. The 38-byte arrays are roughly 2.45 GiB and the mapped page table about 264 MiB before caches, staging, labels, registration, or Vulkan allocation. `cands` alone is roughly 792 MiB even though view collection is capped at 2,048 boxes and a stream batch at 32 (`src/vk/vkbackend.c:5019-5041,5201-5209`).

**Required change:** candidates must be sized to collected visible work, not the entire virtual address space. Pack state into bitsets/smaller fields, use sparse/radix residency metadata and a hierarchical page table, and charge mapped/CPU/GPU metadata to one open-time budget with an actionable rejection/degraded mode.

**Acceptance:** opening the representative manifest uses less than a documented metadata budget (proposed initial target: 512 MiB), initializes in under one warm-filesystem second, and keeps candidate collection p99 below 2 ms without allocating per-virtual-brick candidate objects.

### P0.10 — An accepted LOD manifest can wrap page-table allocation and heap-overflow initialization

Manifest accumulation permits a total page count through `UINT32_MAX` (`src/vk/vkbackend.c:2324-2361`). Renderer initialization later adds the 64-word page header in 32-bit arithmetic (`3948`). For example, four 1,023-cubed grids plus a `15 x 820 x 1,022` grid total 4,294,967,268 pages, which passes the earlier ceiling but wraps `pages + 64` to 36. The allocator creates 36 words (`498-520`), after which initialization clears the 64-word header (`3953`) beyond that allocation.

This is a deterministic heap overflow from a crafted or corrupted manifest. Independently, any near-four-billion-entry page table is far beyond a reasonable metadata budget and should have been rejected before allocation.

**Required change:** checked addition before header inclusion, reject `pages > UINT32_MAX - BR_PAGE_HEADER`, use `size_t` checked byte calculations, and enforce a realistic configured page/metadata budget far below representational maxima. The parser should return the exact violated constraint.

**Acceptance:** the constructed manifest and boundary counts around every addition/product fail cleanly under ASan/UBSan with no allocation or write; property tests prove accepted counts can be safely represented through allocation, initialization, upload, and shader indexing.

### P0.11 — Default host-mapped page publication can expose an atlas mapping before its upload is complete

The default page table is host-mapped and updated by direct CPU stores (`src/vk/vkbackend.c:498-560`). Brick upload submission is asynchronous (`3336-3365`), and the worker transitions the job to done after submission rather than fence completion (`3650-3657`). Main can then publish the new page mapping (`4968-4981`). Eviction drains old readers (`5280-5297`), but a free-slot publication does not establish that the atlas upload has completed before a shader can see the new valid entry. Host writes also need a rigorous frame-reader ownership model; queue order does not automatically order arbitrary host mutation against older/in-flight shader reads.

**Impact:** a page entry can identify brick A while its atlas slot still contains old, partial, or concurrently uploaded bytes.

**Required change:** make page publication a queue-ordered operation after the matching upload, preferably to device-local per-frame/double-buffered page state. Carry `{brick, slot, slot_generation, upload_timeline}` through completion and make the mapping visible only to frames whose dependency includes that value. Do not patch a page table visible to older submissions.

**Acceptance:** slow/canary atlas uploads with rapid free-slot reuse never expose key/data mismatches under synchronization validation and GPU-assisted validation.

### P0.12 — 2x2 view clamping occurs after UBO upload and can permit storage-image writes outside the intended rectangle

`r3d_frame_views` uploads the caller's unclamped `FrameParams` first (`src/vk/vkbackend.c:6584-6587`), then clamps only local dispatch dimensions against the offscreen extent (`6813-6818`). Raycast shader bounds and half-resolution replication still use `pc.viewport` from the uploaded data (`src/shaders/raycast.slang:456-463,504-510`). If resize or a misaligned view leaves a requested viewport larger than the remaining offscreen rectangle, rounded workgroup lanes can pass the shader's larger bound and write beyond the intended storage-image region; half mode can replicate the bad edge.

**Required change:** validate `origin`, `viewport`, additions, and target extent as one rectangle before hashing, UBO upload, and dispatch. Upload the sanitized copy or reject the frame; make the shader independently clamp to the actual target dimensions available through a trusted field/query.

**Acceptance:** explicit origin-near-edge oversized rectangles and rapid resize transitions in full/half modes pass GPU-assisted validation and preserve canary texels surrounding every 2x2 pane.

## P1 findings — scale, persistence, API, and lifecycle

### P1.1 — Whole-plane virtual-slab prefetch is unbudgeted

Each prefetch slot lazily allocates `nx * ny * (D + 2)` and decodes the whole XY plane (`src/vk/vkbackend.c:5631-5645`). Only GPU tile allocation is budget checked (`5935-5941`). With the annotation default of five forward slots plus one behind (`src/main.c:1676,2300-2303`) and an 8,192 x 8,192 x 8 window (`1761-1769`), each slot is about 640 MiB and six slots about 3.75 GiB. A 43,008-square configuration would be about 17.2 GiB per slot.

Use tiled/striped per-Z storage, compressed or mmap-backed windows, a byte-budgeted eviction policy, and automatic slot reduction. Estimate and display the memory plan before allocating; reject configurations that cannot meet the configured ceiling.

### P1.2 — Sparse labels consume dense near-gigabyte state and dirty checks scan it per frame

Label initialization allocates generation arrays for every brick at every LOD and pointer/saved-generation arrays for each base brick (`src/core/labelvol.c:39-60`). At the representative 43,008 x 43,008 x 68,608 scale this is roughly 956 MiB even with no labels. `r3d_labelvol_dirty` scans every base brick (`194-200`) and the label panel calls it during GUI construction (`src/main.c:4549`), amounting to about 60.5 million comparisons and roughly 484 MiB of memory traffic per displayed frame. Save traverses the full grid for one dirty brick (`labelvol.c:230-268`).

Use a sparse brick map, dirty set and O(1) dirty count. Parent generations should be a compact sparse hierarchy. Empty label enablement should be near-constant cost; proposed gates are less than 50 MiB, less than 100 ms enable, and under 10 microseconds for dirty status independent of volume size.

### P1.3 — Decoded caches are duplicated without a process-wide budget

Every `cpuvol` instance eagerly reserves `nslots * 2 MiB` (`src/core/cpuvol.c:90-103`). Hard-coded clients include registration moving 1,024 slots (2 GiB), fixed registration 768 (1.5 GiB), live ink 512 (1 GiB), surface preview 256 (512 MiB), plus predictor and tracer instances (`src/core/regvol.c:168-176,449-460`; `src/core/inklive.c:329-344`; `src/main.c:4971`; `src/core/surfpred.c:109-123`; `src/core/tracer.c:4970,5000`). The same dataset/brick can occupy several caches; registration alone can reach 3.5 GiB.

Use one shared immutable decoded-brick cache keyed by dataset identity, LOD, and coordinate, with pinned leases, a byte cap, shared single-flight I/O, memory-pressure eviction, and per-client priority. Acceptance should run renderer, ink, registration, and tracer concurrently under a fixed total cap and assert that a key has only one resident decoded copy.

### P1.4 — Streaming has large unconditional staging and poorly accounted host memory

With a maximum batch of 32 two-megabyte bricks, the CPU path allocates three heap batches plus three mapped staging batches even without overlays (`src/vk/vkbackend.c:2268-2269,4009-4016`): about 384 MiB before the optional 192 MiB raw network ring (`2573-2591`) and a 256 MiB to 3 GiB warm buffer (`3809-3817`). Decode copies heap -> mapped stage for each lane (`3507-3553`). Warm storage uses the first HOST_VISIBLE|COHERENT type without requiring HOST_CACHED (`src/vk/vkres.c:47-80`) and is later CPU-read for decode, potentially selecting slow write-combined/uncached memory on discrete GPUs.

Allocate only `1 + active_overlays` lanes, budget all pinned/host-visible bytes, prefer ordinary or mmap-backed CPU compressed storage, and select HOST_CACHED upload memory when CPU reads are required. A CT-only configuration should not reserve hundreds of megabytes for inactive overlays.

### P1.5 — The converter can allocate about 33 GiB for a one-job shard

`tools/zarr2c5d/main.c:472-495` allocates a fixed one-gibibyte shard assembly and starts up to 32 phase workers. Each fill worker allocates `chunk_edge^3` bytes before claiming work (`324-340`). The accepted chunk edge reaches 1,024 (`732-735`), so 32 workers can each allocate 1 GiB even though a 1,024-cubed shard contains one source-chunk job: about 33 GiB including assembly. Worker count follows online CPUs, capped only at 32 (`694-697,483-484`), rather than actual jobs and a memory budget.

Cap workers by both job count and peak bytes, use a persistent global pool, claim a job before allocating its scratch, and stream/tile the fixed 1 GiB assembly where practical. Also replace dense one-byte-per-cell surface-selection bitmaps (`537-543`) with sparse ranges/sets for sparse extraction. Resume must validate an existing shard rather than treating any nonempty file as complete (`445`).

### P1.6 — Cache layout scales to millions of inodes

Remote caches store one file per 128-cubed brick in flat LOD directories (`src/vk/vkbackend.c:2439-2442,2653-2680,2783-2839`; CPU analogue `src/core/cpuvol.c:370-415,536`). A million bricks means a million inodes and repeated stat/open/rename operations in one directory; large datasets can reach tens of millions.

Use append-only pack/shard files with an index, negative bitmap, checksummed records, transactional segments, and background compaction. A million cached bricks should require on the order of shards rather than bricks (proposed: fewer than 10,000 inodes) while preserving raw-first first-use latency.

### P1.7 — Several manifest/size parsers disagree or use unchecked arithmetic

The renderer parser has useful bounds (`src/vk/vkbackend.c:2286-2355`), but the CPU parser casts `uint64_t` level dimensions/shard grids to `uint32_t` and accumulates shard counts in unchecked 32-bit arithmetic (`src/core/cpuvol.c:57-88`). `cv_u64_triplet` passes `uint64_t *` as `unsigned long long *` to `sscanf` (`28-34`), a variadic type/portability violation on platforms where the typedefs differ. Mutexes are initialized before many early parse returns and not destroyed on failure (`41-87`). Hash and slot allocation products are not comprehensively checked (`89-100`).

Raw volume open similarly computes `nx * ny * nz` without checked multiplication (`src/core/volume.c:10-43`). TIFF and PNG paths have unchecked dimension products and format-width conversions (`src/core/tifxyz.c:38,48,100-136`; `src/core/pngw.c:39-68`). Registration file reading trusts unchecked `fseek`/`ftell` sizes (`src/core/regvol.c:349-365`).

Create one checked-arithmetic library and one C5D manifest/schema implementation shared by renderer, CPU tools, and converters. Reject unsupported shapes with descriptive limits rather than silent downcasts or generic open failure. Fuzz every parser with ASan/UBSan and boundary values at limit - 1, limit, and limit + 1.

### P1.8 — Tracer export is nontransactional and mutates source state

Five final TIFFs are written sequentially, followed by metadata (`src/core/tracer.c:6514-6594`); a failure can leave a mixed-generation artifact. Optional allocation failures in keep/torn/fill work silently change tear-cut or fill semantics while save may still succeed (`6373-6375,6426-6480`). Fill writes back into `t->pos` during export (`6472-6477`). `fprintf`/`fclose` errors are ignored.

Export from an immutable snapshot into a uniquely named generation directory. Required allocation or write failure aborts the generation; validate checksums and then publish one small atomic pointer/manifest. Source solver state must be byte-identical before and after export.

### P1.9 — Registration and ink artifacts lack transactional and identity guarantees

Registration JSON save writes directly, ignores `fprintf` and `fclose` failures, prints success, and returns zero (`src/core/regvol.c:422-444`). Load uses ad-hoc `strstr`/number extraction (`369-419`), does not reject nonfinite or ill-conditioned transforms, and has no size cap. Transform mutation and generation bumps are separate operations, making it easy for callers to forget invalidation (`116-143`). Refinement size arithmetic from `job_half` is unchecked (`449-452`).

Ink cache validation checks dimensions and `nvalid`, not geometry, CT source, model, or inference/sampling parameters (`src/main.c:65-119`). Same-shaped but different surfaces or model configurations can reuse stale inference. Save removes the valid old target before rename and ignores close/write errors (`76-84`).

Both formats need a common atomic artifact pattern and content identity: source dataset and geometry hash, model/version, parameters, exact dimensions, canonical numeric encoding, checksum, and complete-generation marker.

### P1.10 — Open-data failures and truncation are reported as successful partial listings

The worker ignores `r3d_odlist_fetch` status and publishes normal completion (`src/main.c:1111-1144,1253-1277`). Fetch stops after 32 pages but can still return success, and token/escaping failures also break as if complete (`src/core/odbrowse.c:62-122`). Response bodies are unbounded (`13-27`), and hand XML parsing does not decode entities (`42-53,87-105`). Large buckets can silently omit objects.

Return typed HTTP/CURL/XML/cancel/truncated states and show them in the browser. Use a real streaming XML parser, a response cap, explicit continuation, and a user-visible result count/completeness marker. Test more than 32 pages, malformed XML, escaped keys, HTTP failures, repeated continuation tokens, and cancellation.

### P1.11 — Allocation failure is often converted into success or changed semantics

Boundary-surface candidate/score allocation failure skips growth and can still mark the operation done/successful (`src/core/bsurf.c:512-516,572-596`). Per-generation order allocation failure merely ends work (`550-551`), while offset-cache OOM degrades into "no edge" (`113-182`). Tracer start omits some allocations from its failure validation and grow partially mutates state across allocation failures (`src/core/tracer.c:5789-5818,6197-6245`). Segstore and tracer export have the similar partial-success behavior described above.

Adopt typed internal errors and transactional construction: build complete replacement state, then swap; otherwise leave old state unchanged. Add a deterministic allocator failpoint and run every allocation site in core algorithms through it.

### P1.12 — Live-ink and predictor protocols have weak ownership and wire contracts

`r3d_inklive_poll` exposes module-owned result memory after unlocking; the worker frees/replaces it on the next publication (`src/core/inklive.h:63-65`; `src/core/inklive.c:302-312,387-402`). Poll does not return a complete request/surface/model identity, and the UI accepts results largely by origin (`src/main.c:3237-3258`). Socket writes use plain `write`, risking SIGPIPE; framing and floats use native endianness and received values are not comprehensively checked for finite/range validity (`src/core/inklive.c:41-49,263-300`; `src/core/surfpred.c:43-51`).

Use an ownership-transferring `poll_take` or reference-counted result with a request token and geometry/model hash. Define a versioned endian-stable protocol with bounded lengths, `MSG_NOSIGNAL` or equivalent, exact timeouts/cancellation, and finite/range validation.

### P1.13 — Initialization and lifecycle are not encapsulated

`r3d_tracer_free` neither stops/joins nor destroys its mutex, and its header does not make a precondition explicit (`src/core/tracer.c:6305-6326`; `src/core/tracer.h:268`). `pthread_t` is treated as a Boolean and zeroed, which POSIX does not guarantee (`src/core/tracer.c:6297-6302`; `src/core/bsurf.c:661-671`). Public structs expose mutable buffers, worker, mutex, and CPU-volume internals across tracer, bsurf, inklive, predictor, labels, and segstore. Main even manually constructs tracer state and duplicates invariants (`src/main.c:5117-5151`).

Renderer creation occurs before many later initialization steps that return directly on failure, while comprehensive teardown exists only at the normal end (`src/main.c:1877,1936-2402,6367-6484`). These paths leak resources and can strand threads in sanitizer/failpoint runs.

Make service handles opaque, track explicit lifecycle states and `thread_started`, make destroy idempotently cancel/join/destroy, and centralize application cleanup through one state object. Every initialization stage should be covered by failure injection.

### P1.14 — Command-line parsing silently accepts dangerous values

Main manually rescans options with independent `if` statements, does not consume operands in a central schema, and silently ignores unknown/incomplete options (`src/main.c:1679-1757`). `atoi`, `atof`, and unchecked conversions allow invalid values to become zero or huge unsigned counts; negative frames/sizes can wrap, `--gpu-mem` can overflow its shift, and `--seconds nan` can defeat comparisons and leave a headless run unbounded.

Use `getopt_long` or an equivalent typed parser with complete-consumption checks, finite numeric validation, min/max constraints, unknown-option errors, and generated help. Add CLI tests for missing operands, overflow, negatives, NaN/Inf, duplicate/conflicting options, and typos.

### P1.15 — The current module shape amplifies defect risk

`src/vk/vkbackend.c` is about 7,000 lines and its renderer object owns rendering, swapchain-adjacent state, brick selection, caches, networking, prediction, surfaces, labels, registration, virtual slabs, pipelines, and staging. `src/core/tracer.c` is about 6,800 lines; `src/main.c` about 6,500 lines, with a dataset/session loop spanning most of `main`. Production selftests live inside tracer and bsurf implementation files.

This is not a style-only objection. Ownership rules, error cleanup, cache schemas, and invariants are duplicated because no boundary can be tested or replaced independently. Split by ownership and artifact contract: renderer/WSI, residency scheduler, remote cache, overlay sources, GPU upload/retirement, dataset session, tracing solver, tracer persistence, and QC. Keep APIs narrow, typed, and opaque; move selftests to test binaries.

### P1.16 — Generic image construction infers the wrong Vulkan dimensionality at legal edge sizes

The image helper infers 1D/2D/3D solely from which extent components exceed one (`src/vk/vkres.c:130-188`). Normal surface-volume setup creates a `{1,1,1}` prediction placeholder (`src/vk/vkbackend.c:1805-1812`) and binds it to shader `sampler2D inkpred` (`src/shaders/surfvol.comp:19`), making a 1D view incompatible with the declared descriptor. `r3d_surfvol_begin` accepts one layer (`src/vk/vkbackend.c:1740-1775`), which similarly produces a 2D object where shaders expect 3D. Surface creation permits 1-by-N geometry (`1659-1669`), but shader interpolation uses `dim - 2`/neighbor access and stretch mode needs at least three samples (`src/shaders/surfvol.comp:103-119`; `src/shaders/raycast.slang:545-552`).

Pass image/view dimensionality explicitly rather than inferring it from extent. Use a valid 2D placeholder, enforce at least 2 x 2 interpolation grids and at least three points for stretch, and create a 3D image even when depth is one if the descriptor contract is 3D. Add validation-layer tests for every minimum legal dimension.

### P1.17 — Clip edges, file sizes, and ring depth are not correctly validated

L0/L1 clip requests subtract an apron from a clamped zero origin and then cast the negative coordinate to `uint64_t` (`src/vk/vkclip.c:92-102`; origin clamp in `src/core/clip.h:51-64`). The shard reader sees the huge unsigned coordinate outside the dimension and returns a successful zero-filled region, which is marked valid. Low X/Y clip edges can therefore display false zeros rather than a shifted/clamped edge fill.

Clip pyramid files are mmap'd and `st_size` recorded but never compared with the hard-coded indexed dimensions (`src/vk/vkclip.c:67-88,293-310`); truncated files can cause out-of-bounds access or `SIGBUS`. Public `depth_max` is also not restricted to the fixed 64-element slot arrays (`26-32`); derived loops can exceed them (`src/core/clip.h:68-82`; `src/vk/vkclip.c:166-168,194-197`).

Use signed intersection/clamp plus destination offset or a specified edge-duplication rule, validate exact/required file lengths before mmap use, and reject ring/depth configurations beyond storage. Test the four XY corners at both LODs, truncated pages at every boundary, and depth at 61/62/64/65.

### P1.18 — Ordinary descriptor sets are updated while older submissions may still reference them

Runtime layouts/sets are ordinary descriptor sets without update-after-bind semantics (`src/vk/vkbackend.c:639-709,886-898`). Paths such as CPU-atlas enable and registration/surface binding update those sets without a complete per-frame retirement protocol (`4683-4712,4866-4875`). If an older submission still uses the set, mutation violates Vulkan descriptor lifetime/update rules even when the new resource itself is valid.

Use stable permanent bindings selected through frame data, per-frame descriptor generations, or update-after-bind only with the exact feature/layout/pool flags and synchronization it requires. Retire old sets/resources by timeline. Exercise rapid overlay/registration/surface toggles under synchronization and GPU-assisted validation.

### P1.19 — Clip staging publication has no producer/consumer ownership handoff

Each ring slot has one staging buffer. The worker writes it and atomically publishes done (`src/vk/vkclip.c:121-142`); main later copies from it (`190-232`). The done atomic makes the first payload visible but does not prevent the worker from starting a newer aliased job and overwriting the same bytes while main copies. Recenter cannot cancel a production already underway (`171-187`).

Use a `FREE -> PRODUCING -> READY -> UPLOADING -> FREE` state machine or immutable/double-buffered completed payloads with a request generation. This overlaps the first audit's broader slot-ownership finding but is retained here because the existing atomic protocol may otherwise appear sufficient. Stress rapid recenter/slice changes with delayed producer and consumer plus per-slice checksums under TSan.

## P2 findings — algorithms, portability, and contract drift

### P2.1 — Warm eviction and candidate selection have avoidable superlinear work

Warm allocation uses first-fit free-list scans/memmoves, and each eviction candidate selection scans all warm-resident entries (`src/vk/vkbackend.c:2892-2953`). With many small compressed records, resident count can reach tens or hundreds of thousands; a 32-miss batch can approach `O(32R)`. Candidate submission fully sorts all candidates to consume at most 32 (`5201-5214`).

Use size classes/buddy allocation and heap/clock LRU; use linear top-k/partial selection and sort only the chosen batch. Benchmark adversarial mixed-size churn with at least 100,000 residents and publish p50/p95/p99, fragmentation, and scan counts.

### P2.2 — Surface and segment operations retain large copies or quadratic indexing

TIFF XYZ load holds three float planes and then an interleaved XYZ allocation before freeing planes (`src/core/tifxyz.c:100-136`): roughly 24 bytes per point peak, or about 9.6 GB at 20,000 squared. Stream rows/tiles into the final representation or directly into packed `.tfx` storage, with a bounded external-memory mode.

Segment-store preservation compares every candidate filename against accumulated/old names (`src/core/segstore.c:263-283`), making large rebuilds quadratic. Use a hash index and an append/versioned manifest structure. The fixed 64-cubed occupancy approximation for overlap (`471-523`) should be explicitly documented and tested against thin/curved surfaces with an error bound.

Marching-squares saddle cases 5 and 10 use fixed pairings without an asymptotic/center decider (`src/core/segtrace.c:107-113`), which can connect ambiguous contours incorrectly. Specify the topology policy and add checkerboard/saddle fixtures.

### P2.3 — JSON and binary formats are ad hoc and inconsistent

Umbilicus, TIFF metadata, predictor, tracer, and registration each implement a different substring/`sscanf` style JSON reader (`src/core/umbilicus.c:68-199`; `src/core/tifxyz.c:80-95`; `src/core/surfpred.c:77-106`; `src/core/tracer.c:5856-5873`; `src/core/regvol.c:369-419`). Key matching, whitespace, order, escaping, duplicates, nonfinite numbers, and version policy vary. TIFF loading also changes the process-wide libtiff warning handler (`tifxyz.c:77`).

Adopt one small schema layer with typed getters, limits, duplicate/unknown-field policy, finite checks, and version migrations. For high-volume binary state, use explicit canonical serialization rather than native structs or raw native floats.

### P2.4 — Shader interface/capability drift is not mechanically checked

`r3d_frame_params` is mirrored manually between C and Slang; C asserts only total size 248 (`src/render/render_types.h:94`) while the shader layout is separately declared (`src/shaders/common.slang:5-42`). Size equality does not prove member offsets or flag/constant agreement. Label palettes and related constants are also duplicated.

All raycast variants currently warn that the requested `spirv_1_5` profile is implicitly upgraded (`src/shaders/shaders.cmake:41-43`). The generated SPIR-V validated for Vulkan 1.2 in this review, but the build neither validates SPIR-V nor asserts required capabilities against enabled device features. Comments around raycast brick apron/slot dimensions are stale, and `apron.comp` is still compiled though no runtime use was found.

Generate the shared schema or validate every offset through reflection/golden tests; run `spirv-val` in the shader build/CI; explicitly select the intended target/capabilities and test them against device feature negotiation. Remove or clearly mark dead experimental shaders and stale dimension comments.

### P2.5 — Dependency and platform contracts are only partly reproducible

CMake has strong C5D revision/dirty-tree validation (`CMakeLists.txt:80-119`). The adjacent comment claims the same pinning discipline for fysics, but a local fysics override is checked only for a header, not revision or dirtiness (`121-145`). The Slang fetch script downloads and installs a versioned binary archive without a pinned checksum/signature and removes the old destination before validating the replacement.

Add revision/dirty validation for every local override and checksums/signatures for downloaded toolchains. Either declare Linux/POSIX as the supported platform or isolate `_GNU_SOURCE`, `/proc/self/exe`, pthread, process, and socket assumptions behind a portability layer.

### P2.6 — Several small public APIs rely on unchecked caller invariants

`r3d_tf_build` assumes 1-16 sorted control points and indexes the first/last without validating its public arguments (`src/core/transfer.c:5-20`). PNG and screenshot writers trust dimensions/products and write directly to their final target (`src/core/pngw.c:39-68`; `src/core/screenshot.c:6-27`). Current GUI call sites generally satisfy the transfer-function preconditions, so this is an unsafe reusable contract rather than a reproduced GUI crash.

Validate arguments at API boundaries, use checked products and streaming encoders, and atomically replace output files. Unit-test empty/one/unsorted/duplicate points and zero/overflowing image dimensions.

### P2.7 — Diagnostics do not distinguish unsupported, corrupt, incomplete, and transient data

The renderer's manifest validation is substantially stronger than the CPU parser, but many failures collapse to generic open failure (`src/vk/vkbackend.c:2286-2390,3796-3798`). It ignores levels beyond its maximum rather than clearly rejecting or declaring truncation, and does not fully validate expected halving/shard relations. A failed local reader open can be permanently memoized even if the artifact was merely incomplete/transient.

Return structured reason codes containing dataset, level, coordinate, expected/actual size, retryability, and violated limit. Do not make policy decisions such as permanent negative caching at the lowest I/O layer.

## Test strategy required before broad refactoring

The current CTest surface registers quick, shard safety/data, C5D GPU, and GPU tests (`CMakeLists.txt:259-288`). `tests/test_quick.c` has useful umbilicus/segment geometry coverage, but no direct label-volume, live-ink, surface-predictor, PNG/screenshot, tracer persistence/resume, or CPU-cache concurrency suite. Existing segment tests are primarily happy round trips. The TSan preset passing two quick tests does not exercise the races identified in the first audit. No in-repository CI workflow was found.

Build the safety net before changing architecture:

1. **Artifact crash-consistency harness:** run writers in subprocesses and terminate/fail at every allocation, write, flush, fsync, rename, and manifest-publish step. Assert old complete generation or new complete generation, never a mixture.
2. **Parser fuzzing:** C5D manifests, Blosc HTTP bodies, segment manifests, label bricks/manifests, JSON variants, TIFF dimensions/tags, predictor frames, and CLI input under ASan/UBSan with strict memory caps.
3. **Golden compatibility fixtures:** explicit versions and canonical little-endian artifacts, including legacy migration and byte-swapped/rejected inputs.
4. **Scale tests:** the 43,008 x 43,008 x 68,608 reference geometry, chunk edges 32-1,024, 100,000-segment corpora, one-million-brick caches, sparse empty labels, 20,000-square surfaces, and simultaneous renderer/trace/ink/registration caches. Record peak RSS, allocated bytes by owner, inodes, requests, amplification, and p95/p99 latency.
5. **Failure injection:** deterministic allocator failures, short reads/writes, `ENOSPC`, `EACCES`, corrupt files, malformed/slow servers, rename/fsync failure, missing GPU feature, and worker creation failure.
6. **Sanitizer CI:** release quick, ASan+UBSan core/full, headless TSan lifecycle/cache stress, shader compile + `spirv-val`, and representative Vulkan validation. Fix the current `t.werr` leak before treating this gate as green.
7. **Property tests:** checked arithmetic, coordinate packing uniqueness, save/load identity, cache never publishing unvalidated bytes, and source state unchanged by export.

## Remediation order

1. **Stop silent corruption and loss:** remote transaction validation, label staging/manifest validation, trace save/load fidelity, retain-on-save-failure, predictor nonzero reduction, segstore abort-on-error, and removal of shell interpolation.
2. **Install common safety primitives:** checked arithmetic, bounded buffers, typed errors, unique atomic artifact generations, canonical format headers/checksums, content identity, and failpoints.
3. **Put all memory under one budget:** virtual-brick metadata, decoded caches, remote bodies/chunks/cells, pinned staging, vslab planes, labels, converter scratch, and GPU allocations. Refuse or degrade before allocation.
4. **Consolidate duplicate data paths:** one manifest parser, one remote validator/cache, one decoded-brick cache, one JSON/schema layer, and one artifact publication protocol.
5. **Encapsulate lifecycle and split monoliths:** opaque handles and explicit state transitions first; module moves second. Avoid large mechanical splitting before behavior is captured by tests.
6. **Optimize measured scale ceilings:** sparse metadata/labels, chunk-overlap ingest, packed disk cache, bounded TIFF conversion, nonquadratic segment indexing, and warm-cache/top-k algorithms.
7. **Make quality gates mandatory:** sanitizer/fuzz/fault/scale/shader validation in CI with documented thresholds and artifacts.

## Good patterns to preserve

- Renderer manifest parsing already checks many dimension, grid, and page limits and aligns its 10-bit brick-axis assumptions with shader packing (`src/vk/vkbackend.c:2286-2355`; `src/shaders/surfvol.comp:68-73`). Share and improve it rather than replacing it with another parser.
- The shard reader has strong region bounds/clamping and caches CRC32C validation by immutable file identity (`src/shard/shardio.c:21-83,269-305`).
- Umbilicus load parses into temporary state before swapping, and save already uses `mkstemp`, flush, fsync, and rename (`src/core/umbilicus.c:178-199,223-249`). Add directory durability and async ownership without losing this transactional base.
- TIFF XYZ sanitizes invalid/nonfinite coordinate triples before consumers use them (`src/core/tifxyz.c:117-140`).
- Label initialization cleans partial allocations, and label load sanitizes foreign class IDs (`src/core/labelvol.c:39-65,312-315`).
- Segment-store open performs magic/tile/exact-size and per-segment range checks; its temp+rename helper is a useful starting point (`src/core/segstore.c:32-41,357-393`).
- LOD collection uses distance-shell boxes rather than scanning every virtual brick, and normal stream batches are bounded. The issue is dense auxiliary state and selection/publication cost, not an all-bricks view traversal.
- Network request queues and several worker queues already have deduplication/backpressure; byte budgets, content validation, cancellation, and ownership generations are the missing layers.
- Warning-as-error and ASan/UBSan/TSan configurations exist, C5D pin validation is careful, and existing segment/bsurf geometric selftests provide a useful foundation.

## Limitations

This was an adversarial static and sanitizer/build review, not a production workload benchmark. Memory figures are arithmetic estimates from live constants and representative repository dimensions; they should be confirmed with allocation telemetry and peak-RSS runs. GPU variants validated on the available environment, not a broad vendor/device matrix. External vendored implementation code under `tools/cimgui` and `tools/3ddct` was excluded except at integration/build boundaries.

The first audit remains part of the release assessment. Fixing the defects in this report alone would not resolve the GUI blocking, data races, unsafe worker/source lifetimes, or stale GPU-slot publication documented in `docs/threading-nonblocking-audit-20260819.md`.
