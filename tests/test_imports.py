# -*- coding: utf-8 -*-
"""
Check that every public module of the package can be imported.

A missing or misspelled name in a module-level import statement makes the
whole library unusable, so it is worth catching it explicitly.

"""

import importlib

import pytest

MODULES = [
    "optic.utils",
    "optic.plot",
    "optic.comm.fec",
    "optic.comm.metrics",
    "optic.comm.modulation",
    "optic.comm.ofdm",
    "optic.comm.sources",
    "optic.dsp.carrierRecovery",
    "optic.dsp.clockRecovery",
    "optic.dsp.core",
    "optic.dsp.equalization",
    "optic.dsp.synchronization",
    "optic.models.amplification",
    "optic.models.channels",
    "optic.models.devices",
    "optic.models.perturbation",
    "optic.models.tx",
]

# modules that require CuPy and a CUDA-capable device
GPU_MODULES = [
    "optic.dsp.carrierRecoveryGPU",
    "optic.dsp.coreGPU",
    "optic.models.modelsGPU",
]


@pytest.mark.parametrize("moduleName", MODULES)
def test_module_can_be_imported(moduleName):
    assert importlib.import_module(moduleName) is not None


@pytest.mark.parametrize("moduleName", GPU_MODULES)
def test_gpu_module_can_be_imported(moduleName):
    pytest.importorskip("cupy", reason="CuPy is not installed")

    assert importlib.import_module(moduleName) is not None
