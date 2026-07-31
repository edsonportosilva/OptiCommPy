# -*- coding: utf-8 -*-
"""
Test functions in the optic.dsp.core module.

"""

import numpy as np
import pytest

from optic.dsp.core import (
    decimate,
    delaySignal,
    finddelay,
    firFilter,
    freqShift,
    gaussianNoise,
    lowPassFIR,
    movingAverage,
    phaseNoise,
    pnorm,
    pulseShape,
    quantizer,
    resample,
    sigPow,
    signalPower,
    symbolSync,
    upsample,
)
from optic.utils import parameters


class TestFindDelay:
    def test_finddelay_for_arrays_of_real_values(self):
        delay = 35

        a = np.arange(0, 100)
        b = np.roll(a, -delay)

        assert delay == finddelay(a, b)
        assert delay == -finddelay(b, a)

    def test_finddelay_for_arrays_of_complex_values(self):
        delay = 57

        a = np.arange(0, 100) + 1j * np.arange(0, 100)
        b = np.roll(a, -delay)

        assert delay == finddelay(a, b)
        assert delay == -finddelay(b, a)


class TestPowerNormalization:
    @pytest.mark.parametrize("nModes", [1, 2])
    def test_pnorm_normalizes_the_average_power_per_sample(self, nModes):
        rng = np.random.default_rng(0)
        x = rng.normal(scale=3.0, size=(1000, nModes)) + 1j * rng.normal(
            scale=3.0, size=(1000, nModes)
        )

        y = pnorm(x)

        assert np.mean(np.abs(y) ** 2) == pytest.approx(1.0)

    def test_pnorm_preserves_the_power_ratio_between_modes(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(1000, 2)) + 1j * rng.normal(size=(1000, 2))
        x[:, 1] *= 2

        y = pnorm(x)
        powerIn = np.mean(np.abs(x) ** 2, axis=0)
        powerOut = np.mean(np.abs(y) ** 2, axis=0)

        assert powerOut[1] / powerOut[0] == pytest.approx(powerIn[1] / powerIn[0])

    def test_signalPower_adds_up_the_power_of_all_modes(self):
        x = np.ones((100, 2), dtype=complex)

        assert signalPower(x) == pytest.approx(2.0)

    def test_sigPow_returns_the_average_power_per_mode(self):
        x = 2 * np.ones((100, 2), dtype=complex)

        np.testing.assert_allclose(sigPow(x), 4.0)


class TestFIRFiltering:
    def test_firFilter_with_a_unit_impulse_returns_the_input_signal(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=256) + 1j * rng.normal(size=256)
        h = np.array([0.0, 1.0, 0.0])

        np.testing.assert_allclose(firFilter(h, x), x, atol=1e-12)

    def test_firFilter_compensates_for_the_filter_delay(self):
        # a symmetric filter must not shift the position of a smooth pulse
        n = np.arange(201)
        x = np.exp(-((n - 100) ** 2) / (2 * 5.0**2))
        h = np.ones(11) / 11

        y = firFilter(h, x)

        assert np.argmax(np.abs(y)) == 100

    def test_firFilter_matches_a_direct_convolution(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=512)
        h = rng.normal(size=17)

        np.testing.assert_allclose(
            firFilter(h, x), np.convolve(x, h, mode="same"), atol=1e-10
        )

    def test_firFilter_handles_multimode_signals(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(256, 2)) + 1j * rng.normal(size=(256, 2))
        h = np.array([0.0, 1.0, 0.0])

        y = firFilter(h, x)

        assert y.shape == x.shape
        np.testing.assert_allclose(y, x, atol=1e-12)


class TestPulseShaping:
    @pytest.mark.parametrize("pulseType", ["rect", "nrz", "rrc", "rc"])
    def test_pulseShape_returns_coefficients_with_unit_dc_gain(self, pulseType):
        param = parameters()
        param.pulseType = pulseType
        param.SpS = 8
        param.nFilterTaps = 128

        pulse = pulseShape(param)

        assert pulse.size > 0
        assert np.sum(pulse) == pytest.approx(1.0)

    def test_rrc_pulse_satisfies_the_nyquist_criterion_after_matched_filtering(self):
        # RRC * RRC = RC, which has zero crossings at all non-zero multiples of
        # the symbol period, i.e. the cascade is free of intersymbol interference
        SpS = 8
        param = parameters()
        param.pulseType = "rrc"
        param.SpS = SpS
        param.nFilterTaps = 4096
        param.rollOff = 0.1

        pulse = pulseShape(param)
        rc = np.convolve(pulse, pulse)
        rc = rc / np.max(np.abs(rc))

        center = np.argmax(np.abs(rc))
        offsets = SpS * np.arange(1, 10)

        np.testing.assert_allclose(rc[center + offsets], 0.0, atol=1e-3)
        np.testing.assert_allclose(rc[center - offsets], 0.0, atol=1e-3)


class TestResampling:
    @pytest.mark.parametrize("factor", [2, 4, 8])
    def test_upsample_inserts_zeros_between_the_input_samples(self, factor):
        x = np.array([1.0, 2.0, 3.0, 4.0])

        y = upsample(x, factor)

        assert y.size == x.size * factor
        np.testing.assert_allclose(y[::factor], x)
        assert np.count_nonzero(y) == np.count_nonzero(x)

    def test_decimate_reduces_the_number_of_samples_per_symbol(self):
        rng = np.random.default_rng(3)
        symbols = rng.normal(size=(500, 2)) + 1j * rng.normal(size=(500, 2))
        sig = np.repeat(symbols, 8, axis=0)

        param = parameters()
        param.SpSin = 8
        param.SpSout = 2

        out = decimate(sig, param)

        assert out.shape == (500 * 2, 2)

    def test_resample_changes_the_sampling_rate_by_the_requested_ratio(self):
        rng = np.random.default_rng(4)
        sig = rng.normal(size=1024) + 1j * rng.normal(size=1024)

        param = parameters()
        param.inFs = 4
        param.outFs = 2

        assert resample(sig, param).size == 512


class TestQuantizer:
    # quantizer() indexes x.shape[1], so inputs must be 2D
    @pytest.mark.parametrize("nBits", [2, 4, 6])
    def test_quantizer_produces_at_most_two_to_the_nBits_levels(self, nBits):
        x = np.linspace(-1, 1, 5000).reshape(-1, 1)

        y = quantizer(x, nBits, 1, -1)

        assert np.unique(y).size <= 2**nBits

    def test_quantization_error_is_bounded_by_the_step_size(self):
        nBits = 6
        x = np.linspace(-1, 1, 5000).reshape(-1, 1)

        y = quantizer(x, nBits, 1, -1)
        step = (1 - (-1)) / (2**nBits - 1)

        assert np.max(np.abs(y - x)) <= step / 2 + 1e-12

    def test_quantizer_clips_the_input_to_the_full_scale_range(self):
        x = np.array([[-5.0], [0.0], [5.0]])

        y = quantizer(x, 8, 1, -1)

        assert np.all(y >= -1) and np.all(y <= 1)


class TestLowPassFIR:
    def test_lowpass_filter_passes_dc_and_rejects_the_stopband(self):
        fs, fc, N = 100.0, 10.0, 401

        h = lowPassFIR(fc, fs, N, typeF="rect")
        H = np.fft.rfft(h, 8192)
        freq = np.fft.rfftfreq(8192, d=1 / fs)

        gainDC = np.abs(H[0])
        gainStop = np.max(np.abs(H[freq > 3 * fc]))

        assert gainDC == pytest.approx(1.0, rel=0.05)
        assert gainStop < 0.1 * gainDC


class TestFrequencyShift:
    def test_freqShift_moves_the_spectral_peak_to_the_expected_frequency(self):
        Fs, N, deltaF = 1000.0, 4096, 100.0
        x = np.ones(N, dtype=complex)

        y = freqShift(x, deltaF, Fs)
        freq = np.fft.fftfreq(N, d=1 / Fs)

        assert freq[np.argmax(np.abs(np.fft.fft(y)))] == pytest.approx(
            deltaF, abs=Fs / N
        )

    def test_shifting_forwards_and_backwards_restores_the_input(self):
        rng = np.random.default_rng(5)
        Fs = 1000.0
        x = rng.normal(size=2048) + 1j * rng.normal(size=2048)

        y = freqShift(freqShift(x, 137.0, Fs), -137.0, Fs)

        np.testing.assert_allclose(y, x, atol=1e-9)


class TestMovingAverage:
    def test_moving_average_of_a_constant_signal_is_the_constant(self):
        x = 3 * np.ones((100, 2))

        # the first and last N//2 samples are affected by the zero padding
        np.testing.assert_allclose(movingAverage(x, 5)[2:-2], 3.0)

    def test_moving_average_matches_a_direct_convolution(self):
        rng = np.random.default_rng(6)
        N = 4
        x = rng.normal(size=(64, 1))

        expected = np.convolve(x[:, 0], np.ones(N) / N, mode="same")

        np.testing.assert_allclose(movingAverage(x, N)[10:-10, 0], expected[10:-10])


class TestDelaySignal:
    @pytest.mark.parametrize("delay", [5, 10, 25])
    def test_integer_sample_delay_matches_a_circular_shift(self, delay):
        rng = np.random.default_rng(7)
        x = rng.normal(size=1024)

        y = delaySignal(x, delay, Fs=1)

        assert y.size == x.size
        np.testing.assert_allclose(
            y[100:-100], np.roll(x, delay)[100:-100], atol=1e-9
        )
        assert finddelay(x, y) == -delay

    def test_zero_delay_returns_the_input_signal(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=512)

        # the very last sample is excluded: for delay = 0 no zero padding is
        # added, so the internal np.roll(..., -1) correction drops it
        np.testing.assert_allclose(delaySignal(x, 0, Fs=1)[:-1], x[:-1], atol=1e-9)


class TestNoiseGeneration:
    def test_gaussianNoise_has_the_requested_variance(self):
        noise = gaussianNoise((200000,), 4.0, seed=42)

        assert np.var(noise) == pytest.approx(4.0, rel=0.05)

    def test_gaussianNoise_is_reproducible_with_a_fixed_seed(self):
        a = gaussianNoise((1000,), 1.0, seed=123)
        b = gaussianNoise((1000,), 1.0, seed=123)

        np.testing.assert_array_equal(a, b)

    def test_phaseNoise_increments_follow_the_laser_linewidth(self):
        # random-walk phase noise has increments of variance 2*pi*lw*Ts
        lw, Ts, N = 100e3, 1 / 32e9, 500000

        pn = phaseNoise(lw, N, Ts, seed=11)

        assert np.var(np.diff(pn)) == pytest.approx(2 * np.pi * lw * Ts, rel=0.05)

    def test_phaseNoise_starts_at_zero_phase(self):
        pn = phaseNoise(100e3, 1000, 1 / 32e9, seed=12)

        assert pn[0] == pytest.approx(0.0)


class TestSymbolSync:
    def test_symbolSync_realigns_a_delayed_symbol_sequence(self):
        rng = np.random.default_rng(9)
        SpS = 2
        symbTx = rng.choice([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], size=(2000, 1))
        sigRx = np.roll(np.repeat(symbTx, SpS, axis=0), 17 * SpS, axis=0)

        symbRx = symbolSync(sigRx, symbTx, SpS)

        assert symbRx.shape == symbTx.shape
        assert finddelay(np.abs(sigRx[::SpS, 0]), np.abs(symbRx[:, 0])) == 0
