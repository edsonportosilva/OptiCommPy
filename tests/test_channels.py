# -*- coding: utf-8 -*-
"""
Test functions in the optic.models.channels module.

"""

import numpy as np
import pytest

from optic.dsp.core import (
    finddelay,
    firFilter,
    pulseShape,
    signalPower,
    upsample,
)
from optic.dsp.equalization import edc
from optic.models.channels import awgn, linearFiberChannel, ssfm
from optic.utils import parameters


def awgnParams(snrdB, seed=0, complexNoise=True):
    param = parameters()
    param.snr = snrdB
    param.Fs = 1
    param.B = 1
    param.seed = seed
    param.complexNoise = complexNoise

    return param


class TestAWGN:
    @pytest.mark.parametrize("snrdB", [5, 10, 20, 30])
    def test_output_snr_matches_the_requested_value(self, snrdB):
        # regression test: awgn() reads its configuration through
        # getattr(param, "snr", 20), so it must be given a parameters object
        rng = np.random.default_rng(0)
        sig = rng.normal(size=200000) + 1j * rng.normal(size=200000)

        noise = awgn(sig, awgnParams(snrdB)) - sig
        measured = 10 * np.log10(signalPower(sig) / signalPower(noise))

        assert measured == pytest.approx(snrdB, abs=0.1)

    def test_noise_power_scales_with_the_oversampling_ratio(self):
        rng = np.random.default_rng(1)
        sig = rng.normal(size=100000) + 1j * rng.normal(size=100000)

        param = awgnParams(20)
        param.Fs, param.B = 4, 1

        noise = awgn(sig, param) - sig
        expected = 4 * signalPower(sig) / 10 ** (20 / 10)

        assert signalPower(noise) == pytest.approx(expected, rel=0.02)

    def test_is_reproducible_with_a_fixed_seed(self):
        rng = np.random.default_rng(2)
        sig = rng.normal(size=1000) + 1j * rng.normal(size=1000)

        np.testing.assert_array_equal(
            awgn(sig, awgnParams(15, seed=3)), awgn(sig, awgnParams(15, seed=3))
        )

    def test_real_valued_noise_leaves_a_real_signal_real(self):
        rng = np.random.default_rng(4)
        sig = rng.normal(size=10000)

        out = awgn(sig, awgnParams(20, complexNoise=False))

        assert np.isrealobj(out) or np.allclose(np.imag(out), 0.0)


class TestLinearFiberChannel:
    @pytest.mark.parametrize("L, alpha", [(50, 0.2), (100, 0.2), (80, 0.25)])
    def test_attenuation_follows_the_fiber_loss_coefficient(self, L, alpha):
        rng = np.random.default_rng(5)
        sig = rng.normal(size=8192) + 1j * rng.normal(size=8192)

        param = parameters()
        param.L = L
        param.alpha = alpha
        param.D = 0  # isolate the loss from the dispersion
        param.Fs = 64e9

        out = linearFiberChannel(sig, param)
        lossdB = 10 * np.log10(signalPower(sig) / signalPower(out))

        assert lossdB == pytest.approx(alpha * L, rel=1e-6)

    def test_dispersion_is_energy_preserving(self):
        rng = np.random.default_rng(6)
        sig = rng.normal(size=8192) + 1j * rng.normal(size=8192)

        param = parameters()
        param.L = 200
        param.alpha = 0  # lossless, so only the dispersion acts
        param.D = 16
        param.Fs = 64e9

        assert signalPower(linearFiberChannel(sig, param)) == pytest.approx(
            signalPower(sig), rel=1e-9
        )

    @pytest.mark.parametrize("L", [100, 400])
    def test_chromatic_dispersion_is_undone_by_edc(self, L):
        SpS, Rs = 4, 32e9
        Fs = SpS * Rs

        rng = np.random.default_rng(7)
        symbols = rng.choice([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], size=8192)

        paramPulse = parameters()
        paramPulse.pulseType = "rrc"
        paramPulse.SpS = SpS
        paramPulse.nFilterTaps = 1024
        paramPulse.rollOff = 0.1
        sig = firFilter(pulseShape(paramPulse), upsample(symbols, SpS))

        paramCh = parameters()
        paramCh.L = L
        paramCh.alpha = 0
        paramCh.D = 16
        paramCh.Fc = 193.1e12
        paramCh.Fs = Fs

        paramEDC = parameters()
        paramEDC.L = L
        paramEDC.D = paramCh.D
        paramEDC.Fc = paramCh.Fc
        paramEDC.Fs = Fs
        paramEDC.Rs = Rs

        dispersed = linearFiberChannel(sig, paramCh)
        recovered = edc(dispersed, paramEDC)

        # realign before comparing: the block-wise FFT convolution used by edc()
        # currently leaves a residual delay of a few samples
        recovered = np.roll(recovered, finddelay(np.abs(sig), np.abs(recovered)))

        guard = 2000
        residual = signalPower(
            recovered[guard:-guard] - sig[guard:-guard]
        ) / signalPower(sig[guard:-guard])
        uncompensated = signalPower(
            dispersed[guard:-guard] - sig[guard:-guard]
        ) / signalPower(sig[guard:-guard])

        assert residual < 0.02
        assert residual < uncompensated / 100


class TestSSFM:
    def test_reduces_to_the_linear_channel_when_the_nonlinearity_is_off(self):
        rng = np.random.default_rng(8)
        sig = 1e-3 * (rng.normal(size=4096) + 1j * rng.normal(size=4096))

        paramSSFM = parameters()
        paramSSFM.Ltotal = 80
        paramSSFM.Lspan = 80
        paramSSFM.hz = 1
        paramSSFM.alpha = 0.2
        paramSSFM.D = 16
        paramSSFM.gamma = 0  # switch the Kerr nonlinearity off
        paramSSFM.Fc = 193.1e12
        paramSSFM.Fs = 64e9
        paramSSFM.amp = None
        paramSSFM.prgsBar = False

        paramLin = parameters()
        paramLin.L = paramSSFM.Ltotal
        paramLin.alpha = paramSSFM.alpha
        paramLin.D = paramSSFM.D
        paramLin.Fc = paramSSFM.Fc
        paramLin.Fs = paramSSFM.Fs

        np.testing.assert_allclose(
            ssfm(sig, paramSSFM), linearFiberChannel(sig, paramLin), atol=1e-12
        )

    def test_nonlinearity_broadens_the_signal_spectrum(self):
        # self-phase modulation generates new spectral components
        rng = np.random.default_rng(9)
        sig = rng.normal(size=4096) + 1j * rng.normal(size=4096)

        param = parameters()
        param.Ltotal = 80
        param.Lspan = 80
        param.hz = 1
        param.alpha = 0
        param.D = 0  # isolate the SPM from the dispersion
        param.Fc = 193.1e12
        param.Fs = 64e9
        param.amp = None
        param.prgsBar = False

        param.gamma = 0
        spectrumLinear = np.abs(np.fft.fft(ssfm(sig, param)))

        param.gamma = 1.3
        spectrumNonlinear = np.abs(np.fft.fft(ssfm(sig, param)))

        assert not np.allclose(spectrumLinear, spectrumNonlinear)

    def test_preserves_the_signal_power_without_loss_or_amplification(self):
        rng = np.random.default_rng(10)
        sig = rng.normal(size=4096) + 1j * rng.normal(size=4096)

        param = parameters()
        param.Ltotal = 80
        param.Lspan = 80
        param.hz = 1
        param.alpha = 0
        param.D = 16
        param.gamma = 1.3
        param.Fc = 193.1e12
        param.Fs = 64e9
        param.amp = None
        param.prgsBar = False

        assert signalPower(ssfm(sig, param)) == pytest.approx(
            signalPower(sig), rel=1e-9
        )
