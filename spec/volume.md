# render3d volume conventions (v0, M1)

## Raw volume files (`.u8`)

- Flat array of unsigned 8-bit voxels, x-fastest order: `index = (z*ny + y)*nx + x`.
- No header; dimensions come from the command line (or a `.json` sidecar where one
  exists, e.g. compressor corpus bricks). Single byte per voxel, no endianness issues.
- Coordinate convention matches the c5d corpus: arrays are indexed `[z][y][x]`
  (zarr order); world space maps x→+X, y→+Y, z→+Z with uniform voxel pitch
  (`voxel_um`, informational in M1).

## Spatial hierarchy (inherited from c5d, normative for later milestones)

- 16³ **chunk** — transform/coding unit (c5d).
- 128³ **brick** — random-access / streaming / cache unit. `bricks are the atom of
  residency`: the renderer's future virtual-texture atlas, occupancy structure,
  and LOD selection all operate on bricks, never on whole volumes.
- 1024³ **shard** — file/S3-object unit (`.c5s`, footer-indexed; see
  ~/compressor/spec/format.md).
- LOD levels are fully separate volumes (per-level shards), factor-2 downsampling
  per level.

## M1 dataset

`volume.u8`, 1024³, assembled by `tools/assemble` from the contiguous 8×8×8 brick
grid in `~/compressor/corpus/full` (PHercParis4, chunks z230-237 / y136-143 /
x122-129, 1.129 µm/voxel, level 0). Brick (bz,by,bx) occupies voxels
`[bz*128, ...)` etc., with brick file order also z-fastest-last: brick-local
`index = (z*128 + y)*128 + x`.

## Constraints the renderer must respect

- Never assume a volume fits one GPU texture: `maxImageDimension3D` is 2048 on the
  target driver. M1's 1024³ monolithic texture is a special case; the architecture
  keeps brick metadata in `r3d_volume_desc` from day 1.
- All GPU sampling is normalized-coordinate, R8_UNORM, optimal tiling.
