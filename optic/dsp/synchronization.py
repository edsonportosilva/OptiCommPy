"""
===================================================================
Signal synchronization functions (:mod:`optic.dsp.synchronization`)
===================================================================

.. autosummary::
   :toctree: generated/

   syncDataSequences      -- Synchronize received and transmitted data sequences.
"""

"""Functions for synchronization."""
import logging as logg
import numpy as np

from optic.dsp.core import (
    pnorm,
    firFilter,
    upsample,
    resample,
    symbolSync,
    decimate,
    pulseShape,
)
from optic.comm.modulation import detector, grayMapping
from optic.utils import parameters


def syncDataSequences(rx, tx, param):
    """
    Synchronize data sequences with a given reference.

    Parameters
    ----------
    rx : np.array
        Received signal.
    tx : np.array
        Transmitted reference signal.
    param : optic.utils.parameters object, optional
        Parameters of the synchronization process.

        - param.SpS : samples per symbol of the transmitted signal.
        - param.reference : string ('signal','symbols')
        - param.syncMode : string ('real','amp')
        - param.pulseType : string ('rect','nrz','rrc','rc', 'doubinary')
        - param.rollOff : rolloff of RRC filter. Default is 0.01.
        - param.nFilterTaps : number of filter coefficients. Default is 1024.
        - param.constType : string ('pam','qam','psk')
        - param.M : modulation order.

    Returns
    -------
    np.array
        Synchronized received signal.

    Notes
    -----
    Signals x and y must have the same number of columns (modes).

    """
    SpS = getattr(param, "SpS", 1)
    reference = getattr(param, "reference", "signal")
    syncMode = getattr(param, "syncMode", "real")
    pulseType = getattr(param, "pulseType", "rc")
    rollOff = getattr(param, "rollOff", 0.01)
    nFilterTaps = getattr(param, "nFilterTaps", 1024)
    constType = getattr(param, "constType", "pam")
    M = getattr(param, "M", 4)

    # generate pulse shaping filter
    paramPS = parameters()
    paramPS.pulseType = pulseType
    paramPS.SpS = SpS
    paramPS.rollOff = rollOff
    paramPS.nFilterTaps = nFilterTaps
    pulse = pulseShape(paramPS)

    try:
        rx.shape[1]
        input1D = False
    except IndexError:
        input1D = True
        # If rx is a 1D array, reshape it to a 2D array
        rx = rx.reshape(len(rx), 1)

    try:
        tx.shape[1]
    except IndexError:
        # If tx is a 1D array, reshape it to a 2D array
        tx = tx.reshape(len(tx), 1)

    if reference == "symbols":
        # Upsample transmitted signal
        tx = upsample(tx, SpS)

    # find repetitions of the transmitted signal to match length of received signal
    repeats = np.ceil(rx.shape[0] / tx.shape[0])
    tx_ = np.tile(tx, (int(repeats), 1))

    # calculate required padding
    padL = tx_.shape[0] - rx.shape[0]

    if padL > 0:
        # pad received signal
        rx = np.pad(rx, ((0, padL), (0, 0)))

    # synchronize signals
    tx_ = symbolSync(rx, tx_, 1, mode=syncMode)
    tx_ = tx_[0 : rx.shape[0] - padL, :]

    if reference == "symbols":
        nSymb = int(np.ceil(tx_.shape[0] // SpS) + 1)
        symb = np.zeros((nSymb, tx_.shape[1]))

        for ind in range(tx_.shape[1]):
            outSymb = tx_[tx_[:, ind] != 0, ind]
            symb[0 : len(outSymb), ind] = pnorm(outSymb)  # extract transmitted symbols

        # generate waveform from synchronized symbols
        tx_ = firFilter(pulse, tx_)
        tx_ = pnorm(tx_)
    elif reference == "signal":
        # resample to 41 samples per symbol
        paramRes = parameters()
        paramRes.inFs = SpS  # samples per symbol before resampling
        paramRes.outFs = 41  # samples per symbol after resampling
        x = resample(tx_, paramRes)

        # decimate to symbol rate
        paramDec = parameters()
        paramDec.SpSin = paramRes.outFs
        paramDec.SpSout = 1
        nSymb = np.floor(x.shape[0] // paramDec.SpSin)
        symb = decimate(x[0 : int(nSymb * paramDec.SpSin), :], paramDec)

        # detect symbols
        constSymb = grayMapping(M, constType)
        constSymb = pnorm(constSymb)
        shap = np.shape(symb)
        symb = symb.flatten()
        symb = detector(pnorm(symb), 0.0001, constSymb, rule="ML")[0]
        symb = symb.reshape(shap)
        symb = pnorm(symb)

    if input1D:
        tx_ = tx_.flatten()

    return tx_, symb
