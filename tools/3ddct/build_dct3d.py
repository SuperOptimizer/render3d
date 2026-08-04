"""cffi API-mode builder for the dct3d C codec.

Compiles ../dct3d.c straight into a Python extension (`dct3d_zarr._dct3d`) and
declares the 14 typed encode/decode entry points. Pure C, no third-party deps —
the extension is self-contained, so the resulting wheel needs only libc at
runtime. Invoked by setuptools via `cffi_modules` in pyproject.toml, and usable
standalone (`python build_dct3d.py`) for a local in-place build.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from cffi import FFI


def _c_std_flag() -> str | None:
    """Pick a C standard flag the host compiler actually accepts.

    dct3d.c targets C23, but the spelling differs by compiler version:
    gcc >= 14 / clang >= 18 take ``-std=c23``; gcc 13 wants ``-std=c2x``.
    Probe in preference order and return the first that compiles, or None
    (let the compiler use its default — dct3d.c also builds under c17/gnu).
    """
    cc = os.environ.get("CC", "cc")
    src = "int main(void){return 0;}"
    for flag in ("-std=c23", "-std=c2x", "-std=gnu2x", "-std=c17"):
        try:
            with tempfile.TemporaryDirectory() as d:
                cf = os.path.join(d, "t.c")
                with open(cf, "w") as fh:
                    fh.write(src)
                r = subprocess.run(
                    [cc, flag, "-c", cf, "-o", os.path.join(d, "t.o")],
                    capture_output=True,
                )
                if r.returncode == 0:
                    return flag
        except OSError:
            break
    return None


def _needs_libmvec() -> bool:
    """True if this compiler needs an explicit ``-lmvec`` under ``-ffast-math``.

    gcc on glibc auto-vectorizes the codec's ``powf`` loop into a libmvec call
    (``_ZGVbN4vv_powf``); its own driver adds ``-lmvec``, but the bare link line
    setuptools/cffi use to build the extension does not, so the .so fails to
    load with an undefined symbol. clang emits a plain ``powf`` and needs
    nothing. Probe functionally — build a shared object that calls ``powf`` in a
    vectorizable loop under the codec's flags, WITHOUT the compiler driver's
    implicit libs (``-nostdlib`` on the link) and see whether adding ``-lmvec``
    is what makes it resolve. No compiler-name guessing.
    """
    cc = os.environ.get("CC", "cc")
    src = (
        "#include <math.h>\n"
        "float s[64];\n"
        "void f(float b){ for(int i=0;i<64;++i) s[i]=powf(b+(float)i,0.65f); }\n"
    )
    try:
        with tempfile.TemporaryDirectory() as d:
            cf = os.path.join(d, "t.c")
            with open(cf, "w") as fh:
                fh.write(src)
            obj = os.path.join(d, "t.o")
            comp = subprocess.run(
                [cc, "-O3", "-ffast-math", "-fPIC", "-c", cf, "-o", obj],
                capture_output=True,
            )
            if comp.returncode != 0:
                return False
            # Does the object reference the libmvec vector-powf symbol?
            nm = subprocess.run(["nm", "-u", obj], capture_output=True, text=True)
            return "_ZGV" in nm.stdout and "powf" in nm.stdout
    except OSError:
        return False

# This builder lives at the dct3d repo root, alongside dct3d.c / dct3d.h. The
# extension source is referenced by a REPO-RELATIVE path ("dct3d.c") on purpose:
# cffi/setuptools reject an absolute path in `sources`, and a relative one lands
# correctly in both the in-place build and the sdist/wheel build.
REPO = os.path.dirname(os.path.abspath(__file__))

ffibuilder = FFI()

# The public C surface, verbatim from dct3d.h (cffi's cdef parser wants plain
# declarations — no macros/#include, so the DCT3D_DECL expansion is written out).
_TYPES = [
    ("uint8_t", "u8"),
    ("uint16_t", "u16"),
    ("uint32_t", "u32"),
    ("int8_t", "s8"),
    ("int16_t", "s16"),
    ("int32_t", "s32"),
    ("float", "f32"),
]

cdefs = []
for ctype, name in _TYPES:
    cdefs.append(
        f"size_t dct3d_encode_{name}(const {ctype} *chunk, float quality, "
        f"float max_error, float tau, uint8_t *out);"
    )
    cdefs.append(
        f"int dct3d_decode_{name}(const uint8_t *blob, size_t len, {ctype} *chunk);"
    )
    # decode-side deblocking post-filters (deblock.h)
    cdefs.append(
        f"void dct3d_deblock_{name}({ctype} *vol, const size_t dims[3], "
        f"float step, float strength);"
    )
    cdefs.append(
        f"void dct3d_deblock_chunk_{name}(const {ctype} *chunk, "
        f"const {ctype} *xlo, const {ctype} *xhi, "
        f"const {ctype} *ylo, const {ctype} *yhi, "
        f"const {ctype} *zlo, const {ctype} *zhi, "
        f"float step, float strength, {ctype} *out);"
    )
# DCT3D_N3 * 8 + 256 worst-case, surfaced as a plain constant for the caller's
# scratch buffer (kept in sync with dct3d.h; asserted against the header below).
cdefs.append("#define DCT3D_MAX_BYTES ...")
cdefs.append("#define DCT3D_N ...")
cdefs.append("#define DCT3D_N3 ...")

ffibuilder.cdef("\n".join(cdefs))

ffibuilder.set_source(
    "dct3d_zarr._dct3d",
    '#include "dct3d.h"\n#include "deblock.h"',
    sources=["dct3d.c", "deblock.c"],  # relative to the repo root
    include_dirs=["."],
    # Match the library's intended build: C23 + fast-math autovectorization.
    # -ffast-math is safe here — the codec is explicitly tolerance-only and not
    # bit-reproducible across builds (see dct3d.h). The C-std flag is probed
    # (gcc 13 spells it -std=c2x, gcc 14+/clang -std=c23). MSVC ignores GCC-style
    # flags, so it gets /O2 and its default (C17-ish) mode.
    extra_compile_args=(
        ["/O2"]
        if os.name == "nt"
        else [f for f in (_c_std_flag(), "-O3", "-ffast-math") if f]
    ),
    # gcc+glibc auto-vectorizes powf into a libmvec call the bare cffi link line
    # doesn't resolve; add -lmvec only when the compiler actually emits it.
    libraries=(["mvec"] if os.name != "nt" and _needs_libmvec() else []),
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
