"""
=================================================================
Advanced models for optical transmitters (:mod:`optic.models.tx`)
=================================================================

.. autosummary::
   :toctree: generated/

   simpleWDMTx          -- Implement a simple WDM transmitter.
"""

import numpy as np
from tqdm.notebook import tqdm

from optic.dsp.core import pnorm, pulseShape, signal_power, upsample, phaseNoise
from optic.models.devices import iqm, mzm
from optic.comm.modulation import grayMapping, modulateGray
from optic.comm.sources import symbolSource
from optic.utils import parameters, dBm2W

try:
    from optic.dsp.coreGPU import firFilter
except ImportError:
    from optic.dsp.core import firFilter

import logging as logg


def simpleWDMTx(param):
    """
    Implement a simple WDM transmitter.

    Generates a complex baseband waveform representing a WDM signal with
    arbitrary number of carriers

    Parameters
    ----------
    param : system parameters of the WDM transmitter.
        optic.core.parameter object.

        - param.M: modulation order [default: 16].

        - param.constType: 'qam' or 'psk' [default: 'qam'].

        - param.Rs: carrier baud rate [baud][default: 32e9].

        - param.SpS: samples per symbol [default: 16].

        - param.probabilityDistribution: probability distribution of the symbols [default: 'uniform'].

        - param.shapingFactor: shaping factor of the symbols [default: 0].

        - param.seed: seed for the random number generator [default: None].

        - param.nBits: total number of bits per carrier [default: 60000].

        - param.pulse: pulse shape ['nrz', 'rrc'][default: 'rrc'].

        - param.nPulseTaps: number of coefficients of the rrc filter [default: 4096].

        - param.pulseRollOff: rolloff do rrc filter [default: 0.01].

        - param.mzmScale: MZM modulation scale factor Vrf/Vpi [default: 0.25].

        - param.powerPerChannel: launched power per WDM channel [dBm][default:-3 dBm].

        - param.nChannels: number of WDM channels [default: 5].

        - param.Fc: central frequency of the WDM spectrum [Hz][default: 193.1e12 Hz].

        - param.laserLinewidth: laser linewidth [Hz][default: 100 kHz].

        - param.wdmGridSpacing: frequency spacing of the WDM grid [Hz][default: 40e9 Hz].

        - param.nPolModes: number of polarization modes [default: 1].

    Returns
    -------
    sigTxWDM : np.array
        WDM signal.
    symbTxWDM : np.array
        Array of symbols per WDM carrier.
    param : optic.core.parameter object
        System parameters for the WDM transmitter.

    """
    # check input parameters
    param.M = getattr(param, "M", 16)
    param.constType = getattr(param, "constType", "qam")
    param.Rs = getattr(param, "Rs", 32e9)
    param.SpS = getattr(param, "SpS", 16)
    param.probabilityDistribution = getattr(param, "probabilityDistribution", "uniform")
    param.shapingFactor = getattr(param, "shapingFactor", 0)
    param.seed = getattr(param, "seed", None)
    param.nBits = getattr(param, "nBits", 60000)
    param.pulse = getattr(param, "pulse", "rrc")
    param.nPulseTaps = getattr(param, "nPulseTaps", 4096)
    param.pulseRollOff = getattr(param, "pulseRollOff", 0.01)
    param.mzmScale = getattr(param, "mzmScale", 0.5)
    param.powerPerChannel = getattr(param, "powerPerChannel", -3)
    param.nChannels = getattr(param, "nChannels", 5)
    param.Fc = getattr(param, "Fc", 193.1e12)
    param.laserLinewidth = getattr(param, "laserLinewidth", 0)
    param.wdmGridSpacing = getattr(param, "wdmGridSpacing", 50e9)
    param.nPolModes = getattr(param, "nPolModes", 1)
    param.prgsBar = getattr(param, "prgsBar", True)

    # transmitter parameters
    Ts = 1 / param.Rs  # symbol period [s]
    Fs = 1 / (Ts / param.SpS)  # sampling frequency [samples/s]
    nSymbols = int(param.nBits / np.log2(param.M))  # number of symbols per mode

    # get constellation pmf
    constSymb = grayMapping(param.M, param.constType)
    if param.probabilityDistribution == "uniform":
        px = np.ones(param.M) / param.M
    elif param.probabilityDistribution == "maxwell-boltzmann":
        px = np.exp(-param.shapingFactor * np.abs(constSymb) ** 2)
        px = px / np.sum(px)
    else:
        raise ValueError("Invalid probability distribution.")
    param.pmf = px

    # Symbol source parameters
    paramSymb = parameters()
    paramSymb.nSymbols = param.nBits // int(np.log2(param.M))
    paramSymb.M = param.M
    paramSymb.constType = param.constType
    paramSymb.dist = param.probabilityDistribution
    paramSymb.shapingFactor = param.shapingFactor

    # central frequencies of the WDM channels
    freqGrid = (
        np.arange(-np.floor(param.nChannels / 2), np.floor(param.nChannels / 2) + 1, 1)
        * param.wdmGridSpacing
    )

    if (param.nChannels % 2) == 0:
        freqGrid += param.wdmGridSpacing / 2

    if type(param.powerPerChannel) == list:
        assert (
            len(param.powerPerChannel) == param.nChannels
        ), "list length of power per channel does not match number of channels."
        Pch = (
            10 ** (np.array(param.powerPerChannel) / 10) * 1e-3
        )  # optical signal power per WDM channel
    else:
        Pch = 10 ** (param.powerPerChannel / 10) * 1e-3
        Pch = Pch * np.ones(param.nChannels)

    π = np.pi
    # time array
    t = np.arange(0, int(nSymbols * param.SpS))

    # allocate array
    sigTxWDM = np.zeros((len(t), param.nPolModes), dtype="complex")
    symbTxWDM = np.zeros(
        (len(t) // param.SpS, param.nPolModes, param.nChannels), dtype="complex"
    )

    Psig = 0

    # pulse shaping filter
    if param.pulse == "nrz":
        pulse = pulseShape("nrz", param.SpS)
    elif param.pulse == "rrc":
        pulse = pulseShape(
            "rrc", param.SpS, N=param.nPulseTaps, alpha=param.pulseRollOff, Ts=Ts
        )

    pulse = pulse / np.max(np.abs(pulse))

    if param.seed is not None:
        seed = param.seed
    else:
        seed = None

    for indCh in tqdm(range(param.nChannels), disable=not (param.prgsBar)):
        logg.info(
            "channel %d\t fc : %3.4f THz" % (indCh, (param.Fc + freqGrid[indCh]) / 1e12)
        )

        Pmode = 0
        for indMode in range(param.nPolModes):
            logg.info(
                "  mode #%d\t power: %.2f dBm"
                % (indMode, 10 * np.log10((Pch[indCh] / param.nPolModes) / 1e-3))
            )

            # Generate sequence of constellation symbols
            paramSymb.seed = seed
            symbTx = symbolSource(paramSymb)

            if param.seed is not None:
                seed += 1  # increment seed for next pol/channel

            symbTxWDM[:, indMode, indCh] = symbTx

            # upsampling
            symbolsUp = upsample(symbTx, param.SpS)

            # pulse shaping
            sigTx = firFilter(pulse, symbolsUp)
            sigTx = sigTx / np.max(np.abs(sigTx)) # normalize signal to amplitude 1

            # optical modulation
            if indMode == 0:  # generate LO field with phase noise
                ϕ_pn_lo = phaseNoise(
                    param.laserLinewidth, len(sigTx), 1 / Fs, seed=param.seed
                )
                sigLO = np.exp(1j * ϕ_pn_lo)

            sigTxCh = iqm(sigLO, param.mzmScale * sigTx)
            sigTxCh = np.sqrt(Pch[indCh] / param.nPolModes) * pnorm(sigTxCh)

            sigTxWDM[:, indMode] += sigTxCh * np.exp(
                1j * 2 * π * (freqGrid[indCh] / Fs) * t
            )

            Pmode += signal_power(sigTxCh)

        Psig += Pmode

        logg.info(
            "channel %d\t power: %.2f dBm\n" % (indCh, 10 * np.log10(Pmode / 1e-3))
        )

    logg.info("total WDM signal power: %.2f dBm" % (10 * np.log10(Psig / 1e-3)))

    param.wdmFreqGrid = freqGrid

    return sigTxWDM, symbTxWDM, param


def pamTransmitter(param):
    """
    Generate a optical PAM signal.

    Parameters
    ----------
    param : system parameters of the PAM transmitter.
        optic.core.parameter object.

        - param.M: modulation order [default: 4].

        - param.Rs: symbol rate [baud][default: 32e9].

        - param.SpS: samples per symbol [default: 16].

        - param.probabilityDistribution: probability distribution of the symbols [default: 'uniform'].

        - param.shapingFactor: shaping factor of the symbols [default: 0].

        - param.seed: seed for the random number generator [default: None].

        - param.nBits: total number of bits [default: 40000].

        - param.pulse: pulse shape ['nrz', 'rrc'][default: 'rrc'].

        - param.nPulseTaps: number of coefficients of the rrc filter [default: 4096].

        - param.pulseRollOff: rolloff do rrc filter [default: 0.01].

        - param.power: optical output power [dBm][default:-3 dBm].

        - param.nPolModes: number of polarization modes [default: 1].

    Returns
    -------
    sigTx : np.array
        PAM signal.
    symbTx : np.array
        Array of symbols.
    param : optic.core.parameter object
        System parameters for the PAM transmitter.

    """
    # check input parameters
    param.M = getattr(param, "M", 4)
    param.Rs = getattr(param, "Rs", 32e9)
    param.SpS = getattr(param, "SpS", 16)
    param.probabilityDistribution = getattr(param, "probabilityDistribution", "uniform")
    param.shapingFactor = getattr(param, "shapingFactor", 0)
    param.seed = getattr(param, "seed", None)
    param.nBits = getattr(param, "nBits", 40000)
    param.pulse = getattr(param, "pulse", "rrc")
    param.nPulseTaps = getattr(param, "nPulseTaps", 4096)
    param.pulseRollOff = getattr(param, "pulseRollOff", 0.01)
    param.mzmVpi = getattr(param, "mzmVpi", 3)
    param.mzmVb = getattr(param, "mzmVb", -1.5)
    param.mzmScale = getattr(param, "mzmScale", 0.25)
    param.nPolModes = getattr(param, "nPolModes", 1)
    param.power = getattr(param, "power", -3)
    param.returnParam = getattr(param, "returnParam", False)

    # Symbol source parameters
    paramSymb = parameters()
    paramSymb.nSymbols = param.nBits // int(np.log2(param.M))
    paramSymb.M = param.M
    paramSymb.constType = "pam"
    paramSymb.dist = param.probabilityDistribution
    paramSymb.shapingFactor = param.shapingFactor

    # MZM parameters
    paramMZM = parameters()
    paramMZM.Vpi = param.mzmVpi
    paramMZM.Vb = -param.mzmVb

    # allocate array
    sigTxo = np.zeros(
        ((param.nBits * param.SpS) // int(np.log2(param.M)), param.nPolModes),
        dtype=np.float64,
    )
    symbTx = np.zeros(
        ((param.nBits) // int(np.log2(param.M)), param.nPolModes), dtype=np.float64
    )

    for indMode in range(param.nPolModes):
        if param.seed is not None:
            seed = param.seed + indMode
        else:
            seed = None

        # generate pseudo-random bit sequence
        paramSymb.seed = seed
        symbTx_ = symbolSource(paramSymb)

        # upsampling
        symbolsUp = upsample(symbTx_, param.SpS)

        # pulse shaping filter
        pulse = pulseShape(param.pulse, param.SpS)

        # pulse shaping
        sigTx = firFilter(pulse, symbolsUp)
        sigTx = param.mzmVpi * sigTx / np.max(np.abs(sigTx))

        # optical modulation
        sigTxo_ = mzm(1, param.mzmScale * sigTx, paramMZM)

        # adjust output power
        sigTxo[:, indMode] = dBm2W(param.power) * pnorm(sigTxo_)
        symbTx[:, indMode] = symbTx_

    if param.returnParam:
        return sigTxo, symbTx, param
    else:
        return sigTxo, symbTx
