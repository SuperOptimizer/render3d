# Render3D threading and non-blocking audit

**Date:** 2026-08-19

**Audited revision:** `1c14a98dbfbab54f6607489f85b0375de0f01430`, including the uncommitted worktree present on the audit date

**Scope:** interactive GUI, Vulkan/WSI rendering, 2x2 viewers, volume and brick streaming, downloading, prediction, tracing, registration, labels, surface processing, persistence, lifecycle, and relevant tests

**Method:** adversarial static review by three focused agents plus a cross-cutting architecture review. No product code was changed and no current-worktree performance run was used as evidence.

The audited worktree was already dirty. At audit start, tracked changes existed in `CMakeLists.txt`, `src/core/input.[ch]`, `src/core/tracer.[ch]`, `src/main.c`, `src/render/render.h`, `src/shaders/raycast.slang`, `src/shaders/shaders.cmake`, `src/vk/vkbackend.c`, and `tests/test_quick.c`; new files included `src/core/{bsurf,labelvol,regvol}.[ch]` and `src/shaders/postfilt.comp`. This report describes that live state rather than a clean reconstruction of the commit.

## Executive verdict

Render3D does **not** currently satisfy the requirement that the main GUI thread remain non-blocking and free of heavy work. The application has several good worker-based subsystems, but the main SDL/ImGui loop still performs or waits for substantial CPU, disk, network-adjacent lifecycle, Vulkan, swapchain, and persistence work. Some of those waits are unbounded.

This is not only a responsiveness problem. The review found high-confidence concurrency correctness defects that can publish data into the wrong GPU slot, return CPU cache storage while another thread overwrites it, or let a renderer worker call into a registration object after it has been closed. Those should be treated as release-blocking defects, not deferred performance tuning.

The main conclusions are:

1. The SDL/ImGui thread is also the render/WSI thread. `r3d_frame` can wait indefinitely for a timeline semaphore, swapchain image, or presentation, so a slow GPU/compositor stops event handling (`src/main.c:2424-2429,6274-6278`; `src/vk/vkbackend.c:6466-6475,6511-6515,6936-6961`).
2. Several subsystems do heavy CPU work on workers but synchronously copy, rebuild, upload, wait, or destroy resources on the GUI thread. The handoff therefore still hitches.
3. Cancellation is not end-to-end. Stop and replacement paths commonly set a flag and immediately join, while the worker can be stuck in a blocking network, socket, file, decode, or compute operation.
4. Streaming publication lacks consistent `{resource, slot, generation}` ownership. Fast navigation can make an old completion publish into a physical slot already reassigned to newer data.
5. UI actions directly run whole-grid fitting, voxel painting, cache queries, compression, TIFF/PNG/PPM output, `fsync`, corpus rebuilding, and dataset/overlay initialization.

The appropriate fix is architectural: keep SDL/ImGui as a lightweight event and command producer, give Vulkan and WSI to a dedicated render owner, run CPU/I/O work through bounded cancellable schedulers, publish immutable versioned results, and retire old resources asynchronously after both worker and GPU use have ended.

## Severity and acceptance standard

- **P0 — correctness or responsiveness release blocker:** data race, use-after-free, wrong-resource publication, indefinite/multiminute interactive wait, or unavoidable per-frame GUI blocking.
- **P1 — required for a credible non-blocking claim:** substantial or data-size-dependent GUI work, synchronous lifecycle operation, missing prioritization/cancellation, or routine GPU drain.
- **P2 — hardening/scalability:** contention, oversubscription, avoidable shutdown delay, stale work, or incomplete observability that will become serious at massive-volume scale.

Suggested responsiveness contract:

- SDL event polling and ImGui construction stay on one thread and never call file/network/decode/compression APIs, join workers, wait for GPU primitives, or perform data-size-dependent loops.
- GUI callbacks enqueue commands and return. Target less than 2 ms normally and no callback over 8 ms in an unloaded release build.
- GPU or compositor stalls may reduce visual frame rate, but must not prevent input, window movement, cancellation, progress display, or close handling.
- Under injected 100-500 ms GPU, network, disk, and inference stalls, event-to-visible-UI-response p99 should remain below 50 ms.
- Every queue is bounded, exposes high-water telemetry, has an explicit overflow policy, and supports cancellation or latest-request-wins reconciliation.
- Interactive stop/switch requests return promptly. Cleanup can continue in a reaper state, but the user must not wait for a blocking join.

These numbers are proposed gates, not measurements of the current worktree. They should be adjusted only through an explicit product decision.

## Current execution model

The core loop in `src/main.c:2408-6322` serially performs event polling, navigation, GUI construction, completed-job application, streaming selection/publication, 2x2 overlay preparation, and `r3d_frame`. That makes every synchronous renderer API part of GUI latency. `src/render/render.h:234-241` also documents synchronous rendering behavior rather than an asynchronous command boundary.

| Domain | Existing useful concurrency | Remaining failure mode |
|---|---|---|
| Vulkan rendering | Brick decode, virtual slab, registration atlas, and optional presentation workers exist | Frame wait/acquire/present, many uploads, resize, screenshots, and resource swaps still block the GUI; ownership crosses threads inconsistently |
| Brick/volume streaming | Bounded decode/network queues and a persistent brick worker exist | Legacy slab/clip publication is synchronous or racy; selection and page publication can wait in the main path; obsolete work is weakly cancelled |
| 2x2 viewers | Unchanged panes are cached and collapsed panes are skipped | CPU corpus queries, sorting, tracing, and line construction occur on the GUI thread, often under the cache mutex |
| Tracing/surfaces/ink | Major evolution and inference work is backgrounded | Start/grow/snapshot/fit/stitch/save/upload and stop/join paths remain synchronous; status publication has races |
| Registration/labels | Registration refinement and registration atlas generation use workers | Registration source lifetime and slot ownership are unsafe; label work is intentionally left on the GUI thread and can synthesize/upload millions of voxels per frame |
| Downloading | Data-browser listing and subprocess stdout pumping are background/non-blocking in steady state | Curl/child shutdown and cancellation are incomplete; speculative work can block visible work |
| Persistence | Atomic-renaming patterns are used in several writers | Encoding, directory scanning, full artifact writes, store rebuilds, and `fsync` are frequently initiated and completed inline on the GUI thread |

## P0 findings

### P0.1 — Registration atlas worker outlives its source

The registration atlas worker stores a raw callback/source pointer and calls `gen` and `fetch` while it is alive (`src/vk/vkbackend.c:4653-4694,4702-4723,4763-4777`). The GUI can close or replace `g_reg` directly (`src/main.c:200-215,4665-4670`), and dataset cleanup closes it before brick/renderer teardown (`src/main.c:6465-6473`). `r3d_regvol_close` joins its own job, destroys the mutex, closes storage, frees scratch, and zeroes the object (`src/core/regvol.c:183-193`). The atlas worker is stopped later in renderer cleanup (`src/vk/vkbackend.c:4628-4648`, cleanup callers around `1178-1223`).

**Impact:** use-after-free, destroyed-mutex access, deadlock, crash, or corrupted atlas data during close, replacement, dataset swap, or exit.

**Required change:** add explicit source attach/detach ownership. A detach must stop new atlas requests, cancel or quiesce the current source call, join/reap the atlas worker away from the GUI, and only then release the source. Prefer a refcounted source handle so old and new generations can overlap safely while the old generation retires.

**Gate:** repeatedly attach, close, replace, and swap datasets while source fetch is delayed; pass ASan and headless TSan without lifetime errors or GUI watchdog failures.

### P0.2 — Registration atlas can publish the wrong brick and mutates an in-flight descriptor

The atlas worker reads `r->bs.slot_brick[s]`, then may perform a long demand fetch/resample before uploading (`src/vk/vkbackend.c:4708-4757`). The streaming path rewrites `slot_brick`, `brick_slot`, and `slot_use` separately (`src/vk/vkbackend.c:5176-5221`); the atlas mutex does not protect those writes. A slot can therefore be reassigned while an old atlas request is in flight.

Registration-flat selection can also call ordinary `vkUpdateDescriptorSets` every frame (`src/main.c:6105`; `src/vk/vkbackend.c:4811-4819`; `src/vk/vkres.c:405-414`) while another frame may still reference that descriptor set. The sets are not update-after-bind.

**Impact:** registration data for brick A can be uploaded to a slot now representing brick B; descriptor mutation can cause Vulkan undefined behavior, mixed sampling, validation failures, or GPU faults.

**Required change:** stream immutable residency events containing `{slot, brick, slot_generation}`. Validate the token immediately before submission and publication. Keep both registration images permanently bound and select via frame data, or maintain per-frame descriptor sets updated only after the owning frame retires.

**Gate:** use a one- or two-slot pool, aggressively evict while registration fetch is delayed, and render brick-ID canaries. Rapidly toggle the flattened source under Vulkan synchronization validation.

### P0.3 — CPU-volume cache returns unpinned mutable storage

`cv_brick` returns a pointer into the shared LRU slab after releasing `v->mu` (`src/core/cpuvol.c:570-590,685-707`). Another caller can evict and overwrite that 2 MiB slot (`src/core/cpuvol.c:431-450`) before or during consumption. The TLS memo reads `v->keys[memo_slot]` without the mutex (`src/core/cpuvol.c:578-580`), which is itself a C data race. Consumers dereference after unlock and, in `r3d_cpuvol_read_block`, can copy a large intersection from that pointer (`src/core/cpuvol.c:459-493,710-720`). Close has no active-reader lifetime protocol (`src/core/cpuvol.c:157-177`), and a TLS memo can survive close/reopen at the same object address.

This is a real concurrent path: live-ink fans out as many as 16 samplers (`src/core/inklive.c:106,121-139`), and CPU volumes are explicitly presented as thread-safe (`src/core/cpuvol.h:46-49`).

**Impact:** nondeterministic wrong voxels, corrupted inference/registration results, formal data races, and possible use-after-free during teardown.

**Required change:** return immutable refcounted/pinned brick objects, use epoch/hazard reclamation, or copy from a protected slot into caller-owned scratch before releasing protection. Memo validation and cache lifetime must be synchronized; close must wait for or invalidate active leases without blocking the GUI.

**Gate:** 1-2 cache slots, at least two readers on disjoint bricks, and a concurrent eviction churner; compare hashes under TSan/ASan and repeatedly close/reopen the same object.

### P0.4 — Virtual-slab and clipmap physical slots lack complete generation ownership

Virtual-slab jobs carry coordinates and state but no physical-slot owner generation (`src/vk/vkbackend.c:343-377`). Deduplication compares world coordinates, not toroidal physical slots (`src/vk/vkbackend.c:6167-6207,6210-6232,6255-6279`). Main applies a done job and publishes its old key unconditionally (`src/vk/vkbackend.c:6066-6091`). A rapid pan by more than the grid span can alias old and new world cells to the same tile, allowing an old completion to republish while a newer job overwrites the bytes.

Clipmap ring slots similarly have one shared staging region and request/done/gpu scalars (`src/vk/vkclip.c:26-31`). The worker publishes done and can move on (`src/vk/vkclip.c:121-142`); the GUI-side pump copies from that staging area (`src/vk/vkclip.c:190-234`) while a later aliased job can begin writing it. Z changes do not create a new request generation; only XY recenter does (`src/vk/vkclip.c:151-186`).

**Impact:** data races, wrong slice/cell keys, visible corruption, or GPU sampling partially overwritten staging content during rapid navigation.

**Required change:** every physical tile needs a monotonically increasing owner generation and a state machine covering queued, producing, ready, uploading, published, and retired. Never allow two producer/consumer generations to share mutable staging storage. Discard stale completions, purge obsolete Z/window jobs, and prioritize the newest visible window.

**Gate:** adversarial pans beyond the grid span and random Z on every frame, with per-cell/per-slice canary hashes and TSan.

### P0.5 — The GUI blocks on GPU and WSI every frame

The GUI calls `r3d_frame` directly (`src/main.c:6274-6278`). That call uses an infinite timeline wait for frame-slot reuse (`src/vk/vkbackend.c:6466-6475`) and an infinite swapchain acquire timeout (`src/vk/vkbackend.c:6511-6515`). Presentation is synchronous by default; the presentation thread is opt-in via `R3D_PRESENT_THREAD` (`src/vk/vkbackend.c:1003,6947-6961`). Even with it enabled, the caller waits while its sole prior request is pending/busy (`src/vk/vkbackend.c:6936-6945`). Submit and present also share externally synchronized queue ownership (`src/vk/vkctx.c:329-368`).

**Impact:** a slow GPU, vsync/compositor pause, minimized/occluded surface, driver problem, or queue backlog freezes SDL events and ImGui.

**Required change:** a dedicated render/WSI thread must own acquire, frame-slot retirement, command recording, submission, presentation, swapchain generations, and all Vulkan object mutation. The GUI publishes a latest-wins immutable scene snapshot through a bounded mailbox. If no render slot/image is ready, the renderer skips or waits on its own thread; the GUI continues.

**Gate:** inject stalls into acquire, submit, and present; window movement, input, cancel, progress, and close must remain responsive.

### P0.6 — Stop/switch paths can block for minutes

- `r3d_inklive_stop` sets quit and joins before closing the socket (`src/core/inklive.c:347-365`). Exact blocking I/O can wait 900 seconds for receive and 60 seconds for send (`src/core/inklive.c:21-49`). Dataset swap and exit call stop inline (`src/main.c:2146-2148,6398`).
- Overlay/prediction switching waits for decode state, polls network jobs, drains presentation, and device-idles (`src/vk/vkbackend.c:4324-4383`). Predictor sockets permit 300-second receives (`src/core/surfpred.c:26-40,200-220`).
- `r3d_tracer_stop` sets a plain flag and joins (`src/core/tracer.c:6297-6303`), but underlying CPU-volume calls can spend 30 seconds connecting plus 60 seconds below the low-speed threshold and do not receive tracer cancellation (`src/core/cpuvol.c:288-296`). Many UI paths invoke the stop inline (`src/main.c:4084-4089,4150-4201,4228-4243,4897,5192,5805,6433`).
- Boundary-surface, registration, corpus-cache, data-browser, and virtual-slab cleanup also synchronously join work whose current I/O/decode may not observe the stop flag promptly.

**Impact:** dataset replacement, cancellation, window close, or a normal UI button can freeze from seconds to 15 minutes.

**Required change:** cancellation tokens must propagate through socket `poll` loops, curl progress callbacks, file/decode loops, prediction, prefetch, and GPU queues. `request_stop` should return immediately; a lifecycle coordinator/reaper handles completion. Close/shutdown the socket before join so blocked I/O wakes. Apply deadlines per operation and discard old-generation results.

**Gate:** fake servers that accept and never reply, blackholed HTTP, slow decode, and slow filesystem; stop/switch UI request below 100 ms and bounded background cleanup.

### P0.7 — Tracer and boundary-surface status/cancellation contain C data races

Tracer uses plain cancellation/status/data fields across worker and GUI paths (`src/core/tracer.c:3805-3882,3927-3947,4300-4465,6328-6339`), while the GUI directly reads QC/fit fields (`src/main.c:4041-4043,4142-4149`). Snapshot locking does not establish coherence for all of those writes. Boundary-surface quit is a plain boolean read/written by different threads (`src/core/bsurf.c:516,554,661-668`), and status is written under its mutex but read outside the protected snapshot (`src/core/bsurf.c:561-596`; `src/main.c:3994-3997`).

**Impact:** undefined behavior, missed cancellation, inconsistent progress/QC display, corrupt snapshots, or an indefinitely delayed join.

**Required change:** make cancellation atomic, and publish one immutable status/result generation under a well-defined lock or double buffer. The GUI must not inspect live worker-owned structures directly.

**Gate:** headless TSan loops covering start, grow, anchors, snapshot, completion, stop, and destroy. Avoid using a full SDL desktop process as the only TSan oracle because third-party desktop libraries can add unrelated races.

### P0.8 — Label atlas work is intentionally synchronous on the GUI thread

Main calls `r3d_bricks_labels_sync(..., 8)` every frame (`src/main.c:6091-6092`). Labels are explicitly kept unthreaded because the GUI mutates them (`src/vk/vkbackend.c:205-210,4780-4786`). A coarse label brick synthesizes all `128^3` voxels (`src/core/labelvol.c:151-192`); the sync path scans resident slots and waits/reuses upload resources (`src/vk/vkbackend.c:4702-4757`). Eight coarse bricks can mean roughly 16.8 million voxel iterations plus copies/uploads in one frame.

**Impact:** severe brush/navigation hitches exactly when the user is interacting; merely moving this routine to a worker would introduce label data races.

**Required change:** ordered/coalesced brush commands should create copy-on-write/versioned label bricks. Workers synthesize immutable coarse bricks and publish bounded upload packets to the render owner. The render owner validates label and slot generations before upload.

**Gate:** maximum-radius continuous painting while navigating across dirty coarse bricks, with GUI latency, queue bounds, and generation consistency asserted.

## P1 findings: synchronous interactive work

### P1.1 — Legacy slab and clip streaming still end in main-thread work

Slab navigation directly calls `r3d_slab_window` (`src/main.c:2508-2511,3675-3678`). It waits for all relevant in-flight GPU work, synchronously assembles/uploads changed tile slices, and CPU-downsamples the overview (`src/vk/vkbackend.c:2195-2247`). The staged fallback uses a one-shot fence wait (`src/vk/vkres.c:517-559`). Clip disk decode is backgrounded, but the render path pumps publication (`src/vk/vkbackend.c:6400-6403`), and a slice upload may perform a fenced submission (`src/vk/vkclip.c:190-236`).

The prior measured report already recorded a real 3072-square one-slice scroll at about 11.05 ms average / 14.1 ms p95 and a whole refill maximum of 254 ms (`docs/performance-review-20260806.md`, residual bottleneck 1). `docs/measured.md` also records roughly 95 ms per slice for an 8184-square slab. These historical measurements are consistent with the static call path; they are not new measurements of this worktree.

**Recommendation:** persistent slice producers, persistent staging rings, batched copy command buffers, per-layer timeline publication, and byte/time budgets. Visible work outranks speculative work; old windows are cancelled by generation.

### P1.2 — Dataset open/swap and overlay attach are synchronous transactions

`r3d_bricks_begin` is invoked in the dataset loop (`src/main.c:1964`) and performs manifest/file work, allocations, image/pipeline setup, seeding, decode, cache I/O, and uploads (`src/vk/vkbackend.c:3708-4218`). The coarsest seed path is explicitly multi-second and synchronous (`src/vk/vkbackend.c:4029-4157`). Calling `SDL_PumpEvents` inside this work does not run normal state updates or render a responsive UI.

Dataset replacement tears down and reopens on the same GUI-controlled loop (`src/main.c:6472`). Renderer teardown joins workers and waits for GPU idle before freeing large state (`src/vk/vkbackend.c:1143-1369`). Overlay switching similarly drains workers/network, calls `vkDeviceWaitIdle`, destroys the old atlas, parses and loads the new tree, decodes resident content, writes caches, and uploads (`src/vk/vkbackend.c:4324-4564`). Refilter waits for both a worker condition and a GPU timeline in the button callback (`src/main.c:4481-4486`; `src/vk/vkbackend.c:4591-4606`).

**Recommendation:** asynchronous dataset/overlay generations. Keep the old display live while CPU preparation proceeds. The render owner creates/uploads the new resource generation under a budget and atomically swaps only when minimally usable. Old state retires after worker leases and GPU timelines complete. Refilter becomes a request with progress, fallback quality, and cancellation.

### P1.3 — Resize and common resource updates drain the GPU

- Resize is called directly from the event path (`src/main.c:2433`); swapchain recreation device-idles (`src/vk/vkbackend.c:1480-1486`; `src/vk/vkswap.c:125-127`).
- Surface activation/live previews drain presentation and device-idle before recreate/full uploads (`src/main.c:2597,5052,5371`; `src/vk/vkbackend.c:1679-1720`).
- Ink prediction uploads allocate/copy a whole R32F map, one-shot submit, and infinite-fence-wait; size changes device-idle (`src/main.c:3187-3198,3262-3263`; `src/vk/vkbackend.c:1881-1933`).
- Transfer-function changes device-idle and fence-upload a 256-entry LUT (`src/main.c:2513-2520,3663-3665`; `src/vk/vkbackend.c:1988-1992`). The CPU LUT calculation itself is tiny and is not the problem.
- Screenshot capture device-idles, performs synchronous readback, copies/converts pixels, and writes PPM (`src/main.c:1500-1512,6315-6322`; `src/vk/vkbackend.c:6969-7001`; `src/core/screenshot.c:6-27`).

**Recommendation:** render-thread ownership, persistent upload/readback rings, dirty rectangles/bands, per-frame descriptors/resources, deferred destruction by timeline, debounced resize generations, and a one-pending screenshot policy. A worker waits for screenshot readiness and writes the artifact.

### P1.4 — Label editing and persistence perform data-size-dependent GUI work

Brush input stamps spheres through nested voxel loops and can allocate/zero a 2 MiB brick (`src/main.c:239-255,2958-2978`; `src/core/labelvol.c:83-135`). Radius 32 touches roughly 137,000 voxels per stamp before interpolation between input samples is counted. Dirty-count display scans the brick grid (`src/core/labelvol.c:194-200`; `src/main.c:4549`). Save scans every dirty brick, checks 2 MiB for emptiness, compresses using all cores, and writes/renames files; load enumerates, reads, decodes, and repeatedly scans 2 MiB bricks (`src/core/labelvol.c:208-326`). Buttons and final auto-save call these directly (`src/main.c:4497-4517,4549-4554,6455-6460`).

**Recommendation:** O(1) dirty counters, coalesced stroke queue, copy-on-write brick versions, asynchronous snapshot encode/write/load, and transactional clean marking only when the live generation still matches the saved snapshot.

### P1.5 — Tracing, manual surfaces, ink maps, and corpus rebuilding still burden the GUI

- Tracer construction/grow allocates and copies grids inline (`src/core/tracer.c:5768-5825,6191-6258`; `src/main.c:4201-4211`). Polling can copy full position/state/confidence buffers before the later 400 ms display throttle, then build rows/normals and swap the surface (`src/main.c:5310-5405`; `src/core/tracer.c:6328-6339`).
- Manual surface fitting is `O(grid_nodes * clicks)` (`src/main.c:267-403`) and is recomputed after click edits (`src/main.c:2923-2934,4992-5003,5018-5110`). At configured limits it can approach one billion kernel evaluations.
- Boundary-surface evolution is backgrounded, but up-to-100-MB-plus construction/allocation occurs before worker creation (`src/core/bsurf.c:599-651`; generation limit UI at `src/main.c:3988`).
- Live-ink result handling copies/stitches millions of values, uploads the entire map after each tile, and writes raw/PNG artifacts on the GUI path (`src/main.c:3116-3301,4421-4441`; `src/core/pngw.c:8-74`).
- `r3d_inklive_poll` returns a borrowed `il->res` pointer after unlocking (`src/core/inklive.c:387-402`), while the worker frees/replaces the old result on a later publication (`src/core/inklive.c:302-312`). Current request sequencing reduces the window but does not make the API safe if result publication overlaps main-thread copy/upload; use an ownership transfer or refcounted result lease.
- Save/activate and auto-harvest stop traces, use shell `mkdir`, write TIFFs, rebuild the segment store, and close/reopen the corpus cache from the frame loop (`src/main.c:4242-4268,5181-5244`; `src/core/tracer.c:6362-6552`; `src/core/segstore.c:191-354`).

**Recommendation:** latest-wins/debounced fit jobs with spatial bins, background construction and growth, immutable or dirty-band tracer snapshots, tiled ink assembly and subregion upload, and a transactional artifact/store writer actor. Never close/reopen the live corpus to publish one new segment; publish a new manifest/index snapshot atomically.

### P1.6 — 2x2 viewer CPU overlays run on main and hold cache locks

The 2x2 GPU pane cache is useful: unchanged panes are reused and collapsed views are skipped (`src/vk/vkbackend.c:6694-6805`; `src/main.c:6147-6149`). However, segment/corpus overlay preparation is not comparably isolated. Plane queries scan all segments and their tiles (`src/core/segstore.c:560-587`), near queries allocate, scan, and insertion-sort (`src/core/segstore.c:590-636`), and marching-squares tracing walks rows/cells (`src/core/segtrace.c:103-187`). Main performs query, sorting, up to two traces, and draw preparation while holding `sgc.mu` (`src/main.c:4317-4321,5417-5526`), delaying cache worker publication/eviction.

**Recommendation:** a spatial index and generation-keyed overlay jobs. Pin immutable cache entries briefly, release the mutex, build polylines off-thread, and publish an immutable display list. The GUI should only submit already-built line arrays.

### P1.7 — Streaming selection/publication still has main-path compute and waits

Brick selection walks boxes/cones, builds candidates, and sorts on the render caller (`src/vk/vkbackend.c:4987-5253`), including up to 2048 boxes per view and `qsort`. Eviction/page publication can wait for the timeline (`src/vk/vkbackend.c:5233-5239`), and mapped page-table flush can wait before a full or dirty copy (`src/vk/vkbackend.c:533-556`). This work is smaller than decode but violates the requested rule and scales with viewers/residency.

**Recommendation:** perform desired-set calculation in a stream scheduler from immutable camera/view snapshots. Send prioritized residency deltas to the render owner. Use per-frame/double-buffered page tables or queue-ordered GPU copies so publication never waits on the GUI.

### P1.8 — Network and decode scheduling can priority-invert visible work

Virtual-slab speculative prefetch holds one global serialization mutex over discovery/download/join (`src/vk/vkbackend.c:5536-5541,5606-5628`), while visible fills need the same path (`src/vk/vkbackend.c:5751-5755,5813-5816`). Fetches have retries/timeouts but no cancellation callback (`src/vk/vkbackend.c:5370-5426`). A stale speculative window can therefore delay the current visible request.

CPU-volume misses hold `io_mu` across file and synchronous network/transcode work (`src/core/cpuvol.c:603-662`). Prefetch creates and immediately joins as many as 16 threads (`src/core/cpuvol.c:495-560`); its backoff state is unsynchronized, there is no single-flight, and overlapping writers can share a PID-only temporary cache path (`src/core/cpuvol.c:209-219,302-417`). Shard region calls repeatedly create/join up to 16 nested workers (`src/shard/shardio.c:269-305`), even when invoked by persistent clip/virtual-slab workers.

**Recommendation:** central bounded I/O and decode executors, keyed single-flight futures, visible-over-speculative priority, generation cancellation, unique atomic temp files, and a persistent decode pool. Never hold a global mutex across network I/O. The earlier performance report independently identified persistent shard decode/cache and remote cancellation as next work (`docs/performance-review-20260806.md`, residual bottlenecks 2 and 5).

## P2 findings and hardening

1. **Net-ingest shutdown race and cache state:** `ni.quit` is a plain boolean read in curl/retry paths and written under a different synchronization regime (`src/vk/vkbackend.c:1156-1161,2563-2571,2719-2722`). Make it atomic. The approximately 192 MiB raw ring performs 2 MiB copies under one mutex (`src/vk/vkbackend.c:2573-2609`). State can say raw data is available before encode/write succeeds (`src/vk/vkbackend.c:2826-2839`), creating a retry/failure-state hole after ring eviction.
2. **Corpus-cache stale work:** the segment cache has a bounded, deduplicated LIFO worker (`src/main.c:717-850,912-917`), but rapid navigation can retain obsolete requests and `sgc_close` joins an uncancellable decode (`src/main.c:885-909`; `src/core/segstore.c:402-463`). Add desired-set/generation reconciliation and asynchronous epoch retirement.
3. **Open-data lifecycle:** listing is correctly off-thread (`src/main.c:1093-1183`), but curl has only a connect timeout and no total/low-speed/progress cancellation (`src/core/odbrowse.c:55-127`); shutdown joins it (`src/main.c:6475-6480`). Subprocess stdout is nonblocking in steady state (`src/main.c:1055-1091`), but `popen` does not provide explicit child/process-group cancellation and an active child lacks a clear exit cleanup path. Prefer cancellable libcurl plus `posix_spawn`/PID ownership and nonblocking reap/terminate policy.
4. **Umbilicus persistence:** edits rewrite, flush, `fsync`, and rename the file inline (`src/core/umbilicus.c:223-249`; callers `src/main.c:2810-2828,3439-3449,3731-3739,3927-3931`). Use a debounced snapshot writer and final background flush.
5. **Pipeline/startup work:** renderer startup reads SPIR-V and creates numerous pipelines synchronously after the window exists (`src/vk/vkbackend.c:629-738`; `src/vk/vkres.c:352-383`). Persist the pipeline cache, create optional pipelines lazily on the render owner, and keep a responsive loading state.
6. **Progressive compute backpressure:** surface-volume row bands are genuinely asynchronous (`src/vk/vkbackend.c:6548-6682`), but a full-window dispatch without visibility hints can fill the sole queue. Budget GPU time/rows and use a separate compute queue where the hardware and ownership design permit. Dispatch itself should not be mislabeled as CPU blocking; the problem is backlog feeding the GUI-owned frame waits.
7. **Swapchain lifetime validation:** old swapchain destruction ordering at `src/vk/vkswap.c:77-86` deserves validation review while resize ownership is reworked. Enable synchronization validation and ensure dependent views/resources retire before their owner.

## Target architecture

### Thread and ownership boundaries

| Owner | Responsibilities | Must not do |
|---|---|---|
| GUI/SDL thread | Poll events, update lightweight input/UI state, build ImGui command data, enqueue versioned commands, display immutable progress/results | Vulkan waits/calls, joins, blocking I/O, compression, full-volume/grid/map loops, resource destruction |
| Render/WSI thread | Own all Vulkan objects and descriptor mutation; acquire/record/submit/present; consume bounded upload packets; manage swapchain/resource generations and timeline retirement | Network/file decode, unbounded CPU transforms, waiting while holding application-data locks |
| CPU processing pool | Fit, trace, grid/normal construction, label synthesis, occupancy, decode, resample, corpus queries | Vulkan object mutation, unbounded nested thread creation |
| I/O/network pool | File reads, curl, predictor protocol, download/cache fill, cancellation/deadline handling | Holding shared cache/global locks across I/O; speculative work ahead of visible work |
| Artifact writer | TIFF/PNG/PPM/label/umbilicus/store persistence from immutable snapshots; atomic publication | Reading mutable GUI/worker state after enqueue |
| Lifecycle coordinator/reaper | Cancel old generations, wait for workers/GPU timelines off the GUI, then destroy state | Blocking the GUI while quiescing |

The pools may share implementation threads if profiling warrants it, but ownership and priority classes must remain explicit. Avoid creating fresh thread teams inside jobs; that defeats global bounds and causes oversubscription.

### Command and publication rules

1. Every command carries dataset/resource generation and a cancellation token.
2. Every physical GPU/cache slot has an owner generation. Completion is accepted only when `{resource, slot, generation}` still matches.
3. Worker results are immutable, refcounted, copy-on-write, or transferred by ownership. Never return borrowed mutable storage after unlocking.
4. Interactive camera/brush/slice/resize commands use latest-wins coalescing. Persistence commands remain ordered and durable.
5. Visible requests outrank speculative prefetch. Old-generation work is cancelled or discarded without delaying current visible work.
6. Upload and readback use persistent bounded rings. Render consumes a byte/time budget per frame; overload reduces quality/progress rather than blocking input or growing memory.
7. Old resources are retired only after CPU leases and GPU timeline values both complete. No normal interactive operation needs `vkDeviceWaitIdle`.
8. Status/progress is a coherent immutable snapshot, not ad hoc reads from live worker structures.

## Recommended implementation order

### Phase 0 — Correctness and observability

1. Fix registration attach/detach/source lifetime, slot-generation validation, and descriptor update safety.
2. Fix CPU-volume cache pointer lifetime and close/reopen semantics.
3. Add physical-slot generations/state machines to virtual slab and clipmap.
4. Make quit/generation/status synchronization conforming C atomics or protected snapshots; connect cancellation to network/socket/decode operations.
5. Add thread-ID assertions to known heavy/blocking entry points and telemetry for GUI phase max/p99, queue depth/high-water, cancellation age, upload bytes, and every GPU wait/device-idle.

Do not begin broad threading of labels or other mutable state until snapshot/COW ownership exists; otherwise responsiveness work will add races.

### Phase 1 — Isolate the GUI from Vulkan/WSI

1. Create the render-owner thread and latest-wins scene mailbox.
2. Move acquire, frame waits, recording, submit, present, resize, descriptor changes, and Vulkan destruction to it.
3. Add persistent staging/readback rings and deferred timeline retirement.
4. Convert screenshot, transfer LUT, surface swap, and ink-map upload into render commands.

Exit criterion: injected GPU/compositor stalls no longer stop SDL/ImGui, and no Vulkan wait or `vkDeviceWaitIdle` is reachable from the GUI thread.

### Phase 2 — Async dataset and streaming lifecycle

1. Convert dataset open/swap, overlay attach/switch/refilter, and slab window changes to generation-based state machines.
2. Move desired-set/2x2 visibility selection to the stream scheduler and page-table publication to the render owner.
3. Unify visible and speculative requests under bounded priority/single-flight scheduling.
4. Replace nested shard thread creation with a persistent bounded decode pool and decoded-chunk cache.

Exit criterion: cold open, rapid dataset/overlay swap, slab/clip scrub, and blackholed remote sources retain GUI responsiveness and bounded memory.

### Phase 3 — Processing, annotation, and persistence

1. Implement versioned/COW labels, ordered brush jobs, coarse synthesis, and snapshot persistence.
2. Move manual fitting, tracer create/grow/snapshot transformation, surface construction, and ink stitching to cancellable latest-wins jobs.
3. Move segment queries/traces to immutable overlay-display jobs with a spatial index.
4. Add the artifact/store writer and debounced umbilicus writer.

Exit criterion: maximum configured annotations/surfaces/artifacts cannot create a GUI callback over the agreed budget.

### Phase 4 — Qualification and tuning

1. Tune pool sizes, per-frame upload/compute budgets, cache ceilings, and coalescing from telemetry rather than adding ad hoc threads.
2. Qualify both preferred and fallback Vulkan upload paths and multiple devices.
3. Automate the adversarial gates below in CI/nightly runs.

## Adversarial verification matrix

| Test | Fault/load | Required assertion |
|---|---|---|
| GUI heartbeat | 100-500 ms delay in acquire/present/submit/file/decode/curl/predictor | Input/progress/close continues; event response p99 within contract |
| Registration lifecycle | Rapid attach/close/swap with delayed fetch | No UAF/race; old generation never publishes; ASan/TSan clean |
| Slot ownership | Tiny atlas/rings, pan farther than grid span, random Z | Uploaded canary/hash always matches published `{slot,key,generation}` |
| CPU-volume cache | 1-2 slots, concurrent readers plus eviction churn | Stable checksums; no TSan race or stale TLS pointer |
| Cancellation | Accept-never-reply sockets, blackholed HTTP, slow disk/decode | Stop/switch request returns promptly; cleanup deadline bounded |
| Queue saturation | Continuous navigation/brush/resize/screenshot commands | Bounded memory/queue depth; documented latest-wins/drop behavior |
| Label stress | Radius-32 drag, coarse dirty bricks, concurrent save | No GUI stall; save generation consistency; no paint/fetch race |
| 2x2 stress | Four active panes, large corpus, continuous slice changes | Cached panes remain valid; overlay jobs bounded; no cache-lock convoy |
| Vulkan validation | Resize storms, flat-source toggles, overlay swap, screenshots | Synchronization/lifetime validation clean; no in-use descriptor mutation |
| Lifecycle soak | Repeated open/swap/close during every active worker type | No blocking join on GUI, leak, double free, stale publication, or hang |

The CMake configuration already supports TSan (`CMakeLists.txt:19-35`), but current tests (`CMakeLists.txt:259-288`) focus on quick, shard, and GPU correctness rather than subsystem lifecycle and GUI responsiveness. Add headless concurrency tests so SDL/desktop-library noise does not obscure project-owned races, plus a release-build end-to-end watchdog for event latency.

## Good patterns to preserve

- Normal brick streaming uses persistent decode/network workers and bounded/deduplicated request structures (`src/vk/vkbackend.c:2492-2530,3307-3498`).
- Virtual slab and clip have bounded queues and worker-side disk/decode. Their ownership/publication protocol needs correction; the entire idea should not be discarded.
- The segment cache already moves TIFF decode and full grid/normal preparation off main and has bounded/deduplicated work (`src/main.c:717-850,912-917`).
- Registration refinement computation is backgrounded, and its normal poll path is lightweight. The problem is source/atlas lifetime, not the existence of the worker.
- Live-ink sampling/inference normally runs off-thread and unclaimed requests are coalesced.
- Progressive surface-volume rebuild uses row bands, and the 2x2 renderer caches unchanged panes and skips collapsed views.
- Data-browser listing is backgrounded, and download child stdout is pumped nonblocking in steady state.
- Queue access is externally synchronized in `src/vk/vkctx.c:329-368`, which is necessary when worker and render submissions share a queue.

Worker-side waits are not automatically GUI stalls if a worker exclusively owns the affected state and queues are bounded. Likewise, `vkCmdDispatch` is asynchronous and a 256-entry LUT calculation is appropriately small. The review targets caller waits, unsafe ownership, data-size-dependent GUI work, stale work, and queue backpressure—not “put every function on a new thread.” Offline CLI tools may remain blocking by design; when the GUI launches one, however, it must supervise cancellation and process lifetime asynchronously.

## Audit coverage and limitations

The review followed the interactive call graph through `src/main.c`, `src/render`, `src/vk`, `src/core`, `src/shard`, tests/CMake, and GUI-invoked helper behavior. It specifically searched waits/joins, device/queue idle, blocking network and subprocess calls, per-frame data-size-dependent loops, resource lifetime, queues, cancellation, and mutable cross-thread publication. The three parallel reviews focused respectively on GUI/render/2x2/Vulkan, streaming/download/cache/I/O, and processing/services/lifecycle; duplicate findings were independently corroborated before inclusion.

This is a static audit of the actual dirty worktree, so line numbers refer to that state and may move. It did not claim new performance measurements or prove that every suspected race manifests in a particular driver/configuration. Historical slab timings are explicitly attributed to the existing measured reports. The correctness findings above follow directly from ownership and synchronization visible in the code and should be tested first.

## Final priority list for the implementation agent

1. Registration lifetime, slot generation, and descriptor safety.
2. CPU-volume cache pinning/lifetime.
3. Virtual-slab and clip physical-slot generation protocols.
4. End-to-end cancellable stop/switch and conforming status publication.
5. Dedicated Vulkan/WSI render owner and immutable GUI scene mailbox.
6. Versioned/COW label pipeline.
7. Async dataset/overlay/slab/resource publication and deferred retirement.
8. Async tracing/surfaces/ink/corpus/persistence and 2x2 overlay display lists.
9. Unified bounded visible-first I/O/decode scheduling.
10. Automated latency, lifecycle, sanitizer, and generation-consistency gates.

Until items 1-5 are complete, Render3D should not advertise a non-blocking GUI guarantee. Until items 1-4 are complete, adding more concurrency to mutable data structures is likely to make correctness worse.
