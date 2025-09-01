"""GPU-based digital signal processing utilities."""

import logging as logg

import cupy as cp
import numpy as np
from cupy.fft import fft, fftshift, ifft
from cupyx.scipy import signal


def checkGPU():
    """
    Check if a GPU is available.

    Returns
    -------
    bool
        True if a GPU is available, False otherwise.
    """
    try:
        cp.cuda.device.get_device_id()
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def firFilter(h, x, prec=None):
    """
    Perform FIR filtering and compensate for filter delay.

    Parameters
    ----------
    h : np.array
        Coefficients of the FIR filter (impulse response, symmetric).
    x : np.array
        Input signal.
    prec: cp.dtype
        Size of the complex representation.

    Returns
    -------
    y : np.array
        Output (filtered) signal.

    References
    ----------
    [1] P. S. R. Diniz, E. A. B. da Silva, e S. L. Netto, Digital Signal Processing: System Analysis and Design. Cambridge University Press, 2010.
    """
    try:
        x.shape[1]
        input1D = False
    except IndexError:
        input1D = True
        x = x.reshape(len(x), 1)
    nModes = x.shape[1]

    if prec is None:
        if cp.iscomplexobj(x):
            x_ = cp.asarray(x).astype(cp.complex128)
            h_ = cp.asarray(h).astype(cp.complex128)
        else:
            x_ = cp.asarray(x)
            h_ = cp.asarray(h)
    else:
        x_ = cp.asarray(x).astype(prec)
        h_ = cp.asarray(h).astype(prec)

    y_ = x_.copy()

    for n in range(nModes):
        y_[:, n] = signal.fftconvolve(x_[:, n], h_, mode="same")
    y = cp.asnumpy(y_)
    
    if input1D:
        # If the input was 1D, return a 1D array
        y = y.flatten()

    return y


def blockwiseFFTConv(x, h, NFFT=None, freqDomainFilter=False, prec=None):
    """
    Blockwise convolution in the frequency domain using the overlap-and-save FFT method.

    Parameters
    ----------
    x : ndarray
        Input signal.
    h : ndarray
        Filter impulse response.
    NFFT : int, optional
        FFT size to be used. Must be greater than the length of the filter.
        If None, it will be set to the next power of 2 greater than or equal
        to the length of the filter. Default is None.
    freqDomainFilter : bool, optional
        If True, `h` is assumed to be the frequency response of the filter.
        If False, the FFT of `h` will be computed. Default is False.

    Returns
    -------
    y : ndarray
        The filtered output signal.

    Raises
    ------
    ValueError
        If NFFT is not greater than the length of the filter `h`.

    """

    if prec is None:
        if cp.iscomplexobj(x):
            x_ = cp.asarray(x).astype(cp.complex128)
            h_ = cp.asarray(h).astype(cp.complex128)
        else:
            x_ = cp.asarray(x)
            h_ = cp.asarray(h)
    else:
        x_ = cp.asarray(x).astype(prec)
        h_ = cp.asarray(h).astype(prec)

    sigLen = len(x)  # length of the input signal
    M = len(h)  # length of the filter impulse response
    D = (M - 1) // 2  # filter delay

    if NFFT is None:
        NFFT = 2 ** int(cp.ceil(np.log2(M)))

    if NFFT >= M:
        L = NFFT - M + 1  # block length required
    else:
        logg.error("FFT size is smaller than filter length")

    if freqDomainFilter:
        h_ = cp.pad(
            fftshift(ifft(h_)), (0, L - 1), mode="constant", constant_values=0 + 0j
        )
    else:
        h_ = cp.pad(h_, (0, L - 1), mode="constant", constant_values=0 + 0j)

    H = fft(h_)  # frequency response

    discard = M - 1  # number of samples to be discarded after IFFT (overlap samples)
    numBlocks = int(cp.ceil(sigLen / L))  # total number of FFT blocks to be processed
    padLen = (
        numBlocks * L - sigLen
    )  # pad length necessary to complete an integer number of blocks

    # pad signal with padLen zeros + D zeros (to compensate for filter delay)
    x_ = cp.pad(x_, (0, padLen + D), mode="constant", constant_values=0 + 0j)

    # pre-allocate output
    y = cp.zeros(len(x_), dtype="complex")

    # overlap-and-save blockwise processing
    x_ = cp.pad(x_, (M - 1, 0), mode="constant", constant_values=0 + 0j)

    start_idx = 0
    end_idx = NFFT

    for blk in range(numBlocks):
        X = fft(x_[start_idx:end_idx])
        y_blk = ifft(X * H)
        y[blk * L : (blk + 1) * L] = y_blk[discard:]
        start_idx += L
        end_idx = start_idx + NFFT

    y = cp.asnumpy(y)

    return y[D:-padLen]
