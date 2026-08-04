// ladder.h — (quality, tau) schedule per resolution and pyramid level.
//
// Rule: tau = 2*quality; each coarser level halves both; quality floors at 1
// (tau floors at 2). The L0 quality is chosen by the volume's voxel size (um),
// parsed from the volume name (e.g. "2.400um"). Table given by the project:
//
//   ~1.1um / 0.5um -> 64    (finest)
//   ~2.2um / 2.4um -> 32
//   ~4.3um         -> 16
//   ~7.9um / 9.4um -> 8
//   ~45um          -> 2
//
// Boundaries are picked at geometric midpoints so an arbitrary um maps
// sensibly; unknown/huge resolutions fall to the coarsest bucket.

#ifndef DCT3D_EXPORT_LADDER_H
#define DCT3D_EXPORT_LADDER_H

// L0 quality for a given voxel size in micrometres.
static inline float l0_quality_for_um(double um) {
    if (um <= 1.5) return 64.0f;   // 0.5, ~1.1
    if (um <= 3.2) return 32.0f;   // 2.2, 2.4
    if (um <= 6.0) return 16.0f;   // 4.3
    if (um <= 20.0) return 8.0f;   // 7.9, 9.4
    return 2.0f;                   // 45+
}

// (quality, tau) at a given level, halving from L0 and flooring quality at 1.
static inline void ladder_for_level(double um, int level, float *quality, float *tau) {
    float q = l0_quality_for_um(um);
    for (int i = 0; i < level; ++i) {
        q *= 0.5f;
        if (q < 1.0f) { q = 1.0f; break; }
    }
    if (q < 1.0f) q = 1.0f;
    *quality = q;
    *tau = 2.0f * q;
}

#endif  // DCT3D_EXPORT_LADDER_H
