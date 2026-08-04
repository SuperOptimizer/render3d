"""Shim setup.py so setuptools wires the cffi extension.

All project metadata lives in pyproject.toml; this file exists only to declare
`cffi_modules`, which builds dct3d_zarr._dct3d from ../dct3d.c at wheel time.
"""

from setuptools import setup

setup(cffi_modules=["build_dct3d.py:ffibuilder"])
