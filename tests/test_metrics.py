# -*- coding: utf-8 -*-
"""
Test functions in the optic.comm.metrics module.

"""

import numpy as np
import pytest

from optic.comm.metrics import (
    Qfunc,
    calcEVM,
    calcLLR,
    fastBERcalc,
    monteCarloGMI,
    monteCarloMI,
    theoryBER,
)
from optic.comm.modulation import grayMapping, modulateGray
from optic.dsp.core import pnorm
from optic.models.channels import awgn
from optic.utils import dec2bitarray, parameters


def awgnSymbols(M, snrdB, nSymbols=100000, constType="qam", seed=0):
    """Generate a reference symbol sequence and its noisy version."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, int(np.log2(M)) * nSymbols)
    symbTx = pnorm(modulateGray(bits, M, constType))

    param = parameters()
    param.snr = snrdB
    param.Fs = 1
    param.B = 1
    param.seed = seed

    return symbTx, awgn(symbTx, param)


class TestFastBERcalc:
    @pytest.mark.parametrize("M", [4, 16, 64])
    def test_noiseless_transmission_gives_no_errors(self, M):
        symbTx, _ = awgnSymbols(M, snrdB=40, nSymbols=10000)

        with np.errstate(divide="ignore"):
            BER, SER, SNR = fastBERcalc(symbTx, symbTx, M, "qam")

        assert BER[0] == 0.0
        assert SER[0] == 0.0
        assert SNR[0] > 100  # infinite up to floating-point round-off

    @pytest.mark.parametrize(
        "M, EbN0dB", [(4, 6), (16, 10), (64, 14)]
    )
    def test_measured_ber_follows_the_theoretical_curve(self, M, EbN0dB):
        # awgn() reads its configuration from a parameters object; passing a
        # bare float silently falls back to the default 20 dB SNR
        snrdB = EbN0dB + 10 * np.log10(np.log2(M))
        symbTx, symbRx = awgnSymbols(M, snrdB, nSymbols=100000, seed=M)

        BER = fastBERcalc(symbRx, symbTx, M, "qam")[0][0]

        assert BER == pytest.approx(theoryBER(M, EbN0dB, "qam"), rel=0.2)

    @pytest.mark.parametrize("M", [4, 16])
    def test_ber_decreases_monotonically_with_the_snr(self, M):
        BER = [
            fastBERcalc(*reversed(awgnSymbols(M, snrdB, 50000, seed=1)), M, "qam")[0][0]
            for snrdB in [8, 11, 14, 17]
        ]

        assert np.all(np.diff(BER) <= 0)

    @pytest.mark.parametrize("snrdB", [10, 15, 20])
    def test_estimated_snr_matches_the_channel_snr(self, snrdB):
        # the estimate carries a small positive bias at low SNR, since the
        # received symbols are power-normalized and phase-corrected first
        symbTx, symbRx = awgnSymbols(16, snrdB, nSymbols=100000, seed=7)

        SNR = fastBERcalc(symbRx, symbTx, 16, "qam")[2][0]

        assert SNR == pytest.approx(snrdB, abs=0.5)


class TestTheoryBER:
    @pytest.mark.parametrize("constType", ["qam", "psk", "pam"])
    @pytest.mark.parametrize("M", [4, 16])
    def test_theoretical_ber_decreases_with_the_snr_per_bit(self, M, constType):
        BER = [theoryBER(M, EbN0dB, constType) for EbN0dB in range(2, 16, 2)]

        assert np.all(np.diff(BER) < 0)
        assert np.all(np.array(BER) >= 0)

    def test_higher_order_qam_needs_more_energy_per_bit(self):
        assert theoryBER(4, 10, "qam") < theoryBER(16, 10, "qam")
        assert theoryBER(16, 10, "qam") < theoryBER(64, 10, "qam")


class TestQfunc:
    def test_reference_values(self):
        assert Qfunc(0) == pytest.approx(0.5)
        assert Qfunc(1) == pytest.approx(0.158655, abs=1e-6)
        assert Qfunc(3) == pytest.approx(1.349898e-3, rel=1e-5)

    def test_is_symmetric_about_zero(self):
        for x in [0.5, 1.0, 2.0, 4.0]:
            assert Qfunc(x) + Qfunc(-x) == pytest.approx(1.0)


class TestCalcEVM:
    def test_evm_of_a_perfect_signal_is_zero(self):
        symbTx, _ = awgnSymbols(16, snrdB=40, nSymbols=10000)

        np.testing.assert_allclose(calcEVM(symbTx, 16, "qam", symbTx), 0.0, atol=1e-12)

    @pytest.mark.parametrize("snrdB", [15, 20, 25])
    def test_evm_tracks_the_error_to_signal_power_ratio(self, snrdB):
        # calcEVM returns mean(|error|^2)/mean(|reference|^2), i.e. the squared
        # error vector magnitude, which equals 1/SNR for an AWGN channel
        symbTx, symbRx = awgnSymbols(16, snrdB, nSymbols=100000, seed=3)

        EVM = calcEVM(symbRx, 16, "qam", symbTx)[0]

        assert EVM == pytest.approx(10 ** (-snrdB / 10), rel=0.1)


class TestInformationRates:
    @pytest.mark.parametrize("M", [4, 16])
    def test_gmi_approaches_the_number_of_bits_per_symbol_at_high_snr(self, M):
        symbTx, symbRx = awgnSymbols(M, snrdB=30, nSymbols=50000, seed=5)

        GMI, NGMI = monteCarloGMI(symbRx, symbTx, M, "qam")

        assert GMI[0] == pytest.approx(np.log2(M), rel=1e-3)
        assert NGMI[0] == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.parametrize("M", [4, 16])
    def test_gmi_and_mi_agree_for_gray_mapped_constellations(self, M):
        symbTx, symbRx = awgnSymbols(M, snrdB=18, nSymbols=50000, seed=6)

        GMI = monteCarloGMI(symbRx, symbTx, M, "qam")[0][0]
        MI = monteCarloMI(symbRx, symbTx, M, "qam")[0]

        assert GMI == pytest.approx(MI, abs=0.1)

    @pytest.mark.parametrize("M", [4, 16])
    def test_information_rates_increase_with_the_snr(self, M):
        GMI = [
            monteCarloGMI(*reversed(awgnSymbols(M, snrdB, 20000, seed=8)), M, "qam")[0][
                0
            ]
            for snrdB in [5, 10, 15, 20]
        ]

        assert np.all(np.diff(GMI) > 0)


class TestCalcLLR:
    def test_llr_sign_follows_the_log_p0_over_p1_convention(self):
        M = 4
        const = grayMapping(M, "qam").flatten()
        bitMap = dec2bitarray(np.arange(M), int(np.log2(M)))
        px = np.ones(M) / M

        for indSymb in range(M):
            llr = calcLLR(np.array([const[indSymb]]), 0.01, const, bitMap, px)

            # a positive LLR means the bit is more likely to be 0
            np.testing.assert_array_equal(llr < 0, bitMap[indSymb].astype(bool))

    def test_llr_magnitude_grows_as_the_noise_decreases(self):
        M = 4
        const = grayMapping(M, "qam").flatten()
        bitMap = dec2bitarray(np.arange(M), int(np.log2(M)))
        px = np.ones(M) / M

        reliability = [
            np.mean(np.abs(calcLLR(const[:1], σ2, const, bitMap, px)))
            for σ2 in [1.0, 0.5, 0.1, 0.01]
        ]

        assert np.all(np.diff(reliability) > 0)
