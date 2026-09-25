"""
==========================================================================
DSP algorithms for adaptive filtering (:mod:`optic.dsp.adaptiveFiltering`)
==========================================================================

.. autosummary::
   :toctree: generated/

   coreAdaptEq          -- Adaptive equalizer core processing function.
   coreAdaptEqFD        -- Adaptive equalizer core processing function for frequency domain.
   realValuedDFECore    -- Real-valued decision feedback equalizer core.
   complexValuedDFECore -- Complex-valued decision feedback equalizer core.
   realValuedFFECore    -- Real-valued feedforward equalizer core.
   complexValuedFFECore -- Complex-valued feedforward equalizer core.
   volterraCore         -- Volterra equalizer core.
"""

"""Functions for adaptive and static equalization."""
import logging as logg

import numpy as np
from numba import njit
from numpy.fft import fft, ifft
from tqdm.notebook import tqdm

# try:
#     from optic.dsp.coreGPU import blockwiseFFTConv
# except ImportError:
#     from optic.dsp.core import blockwiseFFTConv


@njit(fastmath=True, cache=True)
def coreAdaptEq(
    sigIn,
    symbRef,
    SpS,
    H,
    H_,
    L,
    mu,
    lambdaRLS,
    nTaps,
    storeCoeff,
    runWL,
    alg,
    constSymb,
    prec,
):
    """
    Adaptive equalizer core processing function

    Parameters
    ----------
    sigIn : np.array
        Input signal array.
    symbRef : np.array
        Reference symbol sequence.
    SpS : int
        Samples per symbol.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    L : int
        Length of the output.
    mu : float
        Step size parameter.
    lambdaRLS : float
        RLS forgetting factor.
    nTaps : int
        Number of taps.
    storeCoeff : bool
        Flag indicating whether to store coefficient matrices.
    runWL : bool
        Run widely-linear mode
    alg : str
        Equalizer algorithm.
    constSymb : np.array
        Constellation symbols.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    sigOut : np.array
        Equalized output array.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    errSq : np.array
        Squared absolute error array.
    Hiter : np.array
        History of coefficient matrices.

    """
    # allocate variables
    nModes = int(sigIn.shape[1])
    indTaps = np.arange(0, nTaps)
    indMode = np.arange(0, nModes)

    errSq = np.empty((nModes, L))
    sigIn = sigIn.astype(prec)
    H = H.astype(prec)
    H_ = H_.astype(prec)

    sigOut = sigIn[:L].copy()
    sigOut[:] = np.nan
    outEq = np.array([[0 + 1j * 0]]).repeat(nModes).reshape(nModes, 1).astype(prec)

    if storeCoeff:
        Hiter = (
            np.array([[0 + 1j * 0]])
            .repeat((nModes**2) * nTaps * L)
            .reshape(nModes**2, nTaps, L)
            .astype(prec)
        )
    else:
        Hiter = (
            np.array([[0 + 1j * 0]])
            .repeat((nModes**2) * nTaps)
            .reshape(nModes**2, nTaps, 1)
            .astype(prec)
        )
    if alg == "rls":
        Sd = np.eye(nTaps, dtype=prec)
        a = Sd.copy()
        for _ in range(nTaps - 1):
            Sd = np.concatenate((Sd, a))
    # Radii cma, rde
    Rcma = (
        np.mean(np.abs(constSymb) ** 4) / np.mean(np.abs(constSymb) ** 2)
    ) * np.ones((1, nModes)).astype(prec)
    Rrde = np.unique(np.abs(constSymb)).astype(prec)

    for ind in range(L):
        outEq[:] = 0

        indIn = indTaps + ind * SpS  # simplify indexing and improve speed

        # pass signal sequence through the equalizer:
        for N in range(nModes):
            inEq = sigIn[indIn, N : N + 1]  # slice input coming from the Nth mode
            outEq += (
                H[indMode + N * nModes, :] @ inEq
            )  # add contribution from the Nth mode to the equalizer's output
            if runWL:
                outEq += H_[indMode + N * nModes, :] @ inEq.conjugate()
                # add augmented contribution from the Nth mode to the equalizer's output

        sigOut[ind, :] = outEq.T

        # update equalizer taps acording to the specified
        # algorithm and save squared error:
        if alg == "nlms":
            H, H_, errSq[:, ind] = nlmsUp(
                sigIn[indIn, :], symbRef[ind, :], outEq, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "cma":
            H, H_, errSq[:, ind] = cmaUp(
                sigIn[indIn, :], Rcma, outEq, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "dd-lms":
            H, H_, errSq[:, ind] = ddlmsUp(
                sigIn[indIn, :], constSymb, outEq, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "rde":
            H, H_, errSq[:, ind] = rdeUp(
                sigIn[indIn, :], Rrde, outEq, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "da-rde":
            H, H_, errSq[:, ind] = dardeUp(
                sigIn[indIn, :], symbRef[ind, :], outEq, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "rls":
            H, Sd, errSq[:, ind] = rlsUp(
                sigIn[indIn, :], symbRef[ind, :], outEq, lambdaRLS, H, Sd, nModes, prec
            )
        elif alg == "dd-rls":
            H, Sd, errSq[:, ind] = ddrlsUp(
                sigIn[indIn, :], constSymb, outEq, lambdaRLS, H, Sd, nModes, prec
            )
        elif alg == "static":
            errSq[:, ind] = errSq[:, ind - 1]
        else:
            raise ValueError(
                "Equalization algorithm not specified (or incorrectly specified)."
            )
        if storeCoeff:
            Hiter[:, :, ind] = H
        else:
            Hiter[:, :, 0] = H

    return sigOut, H, H_, errSq, Hiter


@njit(fastmath=True)
def nlmsUp(sigIn, symbRef, outEq, mu, H, H_, nModes, runWL, prec):
    """
    Coefficient update with the NLMS algorithm.

    Parameters
    ----------
    sigIn : np.array
        Input signal array.
    symbRef : np.array
        Reference symbol sequence.
    outEq : np.array
        Equalized output array.
    mu : float
        Step size for the update.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec: data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H : np.array
        Updated augmented coefficient matrix.
    errSq : np.array
        Squared absolute error.

    """
    indMode = np.arange(0, nModes)
    err = symbRef - outEq.T  # calculate output error for the NLMS algorithm

    errDiag = np.diag(err[0]).astype(prec)  # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing and improve speed
        inAdapt = sigIn[:, N].T / np.linalg.norm(sigIn[:, N]) ** 2  # NLMS normalization
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(sigIn), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * errDiag @ inAdaptPar.conjugate()
        )  # gradient descent update
        if runWL:
            H_[indUpdTaps, :] += mu * errDiag @ inAdaptPar  # gradient descent update
    return H, H_, np.abs(err) ** 2


@njit(fastmath=True)
def rlsUp(sigIn, symbRef, outEq, λ, H, Sd, nModes, prec):
    """
    Coefficient update with the RLS algorithm.

    Parameters
    ----------
    sigIn : np.array
        Input signal array.
    symbRef : np.array
        Reference symbol sequence.
    outEq : np.array
        Equalized output array.
    λ : float
        Forgetting factor.
    H : np.array
        Coefficient matrix.
    Sd : np.array
        Inverse correlation matrix.
    nModes : int
        Number of modes.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    Sd : np.array
        Updated inverse correlation matrix.
    errSq : np.array
        Squared absolute error.

    """
    nTaps = H.shape[1]
    indMode = np.arange(0, nModes)
    indTaps = np.arange(0, nTaps)
    Sd = Sd.astype(prec)

    err = symbRef - outEq.T  # calculate output error for the NLMS algorithm

    errDiag = np.diag(err[0]).astype(prec)  # define diagonal matrix from error array

    # update equalizer taps
    Sd = Sd.astype(prec)

    for N in range(nModes):
        indUpdModes = indMode + N * nModes
        indUpdTaps = indTaps + N * nTaps

        Sd_ = Sd[indUpdTaps, :]

        inAdapt = sigIn[:, N].conjugate().reshape(-1, 1).astype(prec)  # input samples
        inAdaptPar = ((inAdapt.T).repeat(nModes).reshape(len(sigIn), -1).T).astype(
            prec
        )  # expand input to parallelize tap adaptation

        A = (Sd_ @ inAdapt).astype(prec)
        B = (inAdapt.conjugate().astype(prec).T @ Sd_).astype(prec)
        C = (inAdapt.conjugate().astype(prec).T @ A).astype(prec)
        num = (A @ B).astype(prec)

        Sd_ = ((1 / λ) * (Sd_ - num / (λ + C))).astype(prec)

        Y = (Sd_ @ inAdaptPar.T).astype(prec).T

        H[indUpdModes, :] += errDiag @ Y

        Sd[indUpdTaps, :] = Sd_
    return H, Sd, np.abs(err) ** 2


@njit(fastmath=True)
def ddlmsUp(sigIn, constSymb, outEq, mu, H, H_, nModes, runWL, prec):
    """
    Coefficient update with the DD-LMS algorithm.

    Parameters
    ----------
    sigIn : np.array
        Input signal array.
    constSymb : np.array
        Array of constellation symbols.
    outEq : np.array
        Equalized output array.
    mu : float
        Step size for the update.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    errSq : np.array
        Squared absolute error.

    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    decided = np.zeros(outEq.shape, dtype=prec)
    sigIn = sigIn.astype(prec)

    for k in range(nModes):
        indSymb = np.argmin(np.abs(outEq[0, k] - constSymb))
        decided[0, k] = constSymb[indSymb]
    err = decided - outEq  # calculate output error for the DDLMS algorithm

    err = err.astype(prec)
    errDiag = np.diag(err[0])  # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = sigIn[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(sigIn), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * errDiag @ inAdaptPar.conjugate()
        )  # gradient descent update
        if runWL:
            H_[indUpdTaps, :] += mu * errDiag @ inAdaptPar  # gradient descent update
    return H, H_, np.abs(err) ** 2


@njit(fastmath=True)
def ddrlsUp(sigIn, constSymb, outEq, λ, H, Sd, nModes, prec):
    """
    Coefficient update with the DD-RLS algorithm.

    Parameters
    ----------
    sigIn : np.array
        Input signal array.
    constSymb : np.array
        Array of constellation symbols.
    outEq : np.array
        Equalized output array.
    λ : float
        Forgetting factor.
    H : np.array
        Coefficient matrix.
    Sd : np.array
        Inverse correlation matrix.
    nModes : int
        Number of modes.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    Sd : np.array
        Updated inverse correlation matrix.
    errSq : np.array
        Squared absolute error.

    """
    nTaps = H.shape[1]
    indMode = np.arange(0, nModes)
    indTaps = np.arange(0, nTaps)

    outEq = outEq.T
    decided = np.zeros(outEq.shape, dtype=prec)

    for k in range(nModes):
        indSymb = np.argmin(np.abs(outEq[0, k] - constSymb))
        decided[0, k] = constSymb[indSymb]
    err = decided - outEq  # calculate output error for the DDLMS algorithm

    errDiag = np.diag(err[0])  # define diagonal matrix from error array

    # update equalizer taps
    Sd = Sd.astype(prec)

    for N in range(nModes):
        indUpdModes = indMode + N * nModes
        indUpdTaps = indTaps + N * nTaps

        Sd_ = Sd[indUpdTaps, :]

        inAdapt = sigIn[:, N].conjugate().reshape(-1, 1).astype(prec)  # input samples
        inAdaptPar = ((inAdapt.T).repeat(nModes).reshape(len(sigIn), -1).T).astype(
            prec
        )  # expand input to parallelize tap adaptation

        A = (Sd_ @ inAdapt).astype(prec)
        B = (inAdapt.conjugate().astype(prec).T @ Sd_).astype(prec)
        C = (inAdapt.conjugate().astype(prec).T @ A).astype(prec)
        num = (A @ B).astype(prec)

        Sd_ = ((1 / λ) * (Sd_ - num / (λ + C))).astype(prec)

        Y = (Sd_ @ inAdaptPar.T).astype(prec).T

        H[indUpdModes, :] += errDiag @ Y

        Sd[indUpdTaps, :] = Sd_
    return H, Sd, np.abs(err) ** 2


@njit(fastmath=True)
def cmaUp(sigIn, R, outEq, mu, H, H_, nModes, runWL, prec):
    """
    Coefficient update with the CMA algorithm.

    Parameters
    ----------
    sigIn : np.array
        Input signal array.
    R : np.array
        Correlation array.
    outEq : np.array
        Equalized output array.
    mu : float
        Step size parameter.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    errSq : np.array
        Squared absolute error.

    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    err = R - np.abs(outEq) ** 2  # calculate output error for the CMA algorithm
    err = err.astype(prec)

    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = sigIn[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(sigIn), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ inAdaptPar.conjugate()
        )  # gradient descent update
        if runWL:
            H_[indUpdTaps, :] += mu * prodErrOut @ inAdaptPar  # gradient descent update
    return H, H_, np.abs(err) ** 2


@njit(fastmath=True)
def rdeUp(sigIn, R, outEq, mu, H, H_, nModes, runWL, prec):
    """
    Coefficient update with the RDE algorithm.

    Parameters
    ----------
    sigIn : np.array
        Input signal array.
    R : np.array
        Constellation radius array.
    outEq : np.array
        Equalized output array.
    mu : float
        Step size parameter.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    errSq : np.array
        Squared absolute error.

    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    decidedR = np.zeros(outEq.shape, dtype=prec)

    # find closest constellation radius
    for k in range(nModes):
        indR = np.argmin(np.abs(R - np.abs(outEq[0, k])))
        decidedR[0, k] = R[indR]
    err = (
        decidedR**2 - np.abs(outEq) ** 2
    )  # calculate output error for the RDE algorithm

    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = sigIn[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(sigIn), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ inAdaptPar.conjugate()
        )  # gradient descent update
        if runWL:
            H_[indUpdTaps, :] += mu * prodErrOut @ inAdaptPar  # gradient descent update

    return H, H_, np.abs(err) ** 2


@njit(fastmath=True)
def dardeUp(sigIn, ref, outEq, mu, H, H_, nModes, runWL, prec):
    """
    Coefficient update with the data-aided RDE algorithm.

    Parameters
    ----------
    sigIn : np.array
        Input signal array.
    ref : np.array
        Reference symbol sequence.
    outEq : np.array
        Equalized output array.
    mu : float
        Step size parameter.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    errSq : np.array
        Squared absolute error.

    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    decidedR = np.zeros(outEq.shape, dtype=prec)

    # find exact constellation radius
    for k in range(nModes):
        decidedR[0, k] = np.abs(ref[k])
    err = (
        decidedR**2 - np.abs(outEq) ** 2
    )  # calculate output error for the RDE algorithm

    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = sigIn[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(sigIn), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ inAdaptPar.conjugate()
        )  # gradient descent update
        if runWL:
            H_[indUpdTaps, :] += mu * prodErrOut @ inAdaptPar  # gradient descent update
    return H, H_, np.abs(err) ** 2


@njit(fastmath=True, cache=True)
def coreAdaptEqBlock(
    sigIn,
    symbRef,
    SpS,
    H,
    H_,
    L,
    mu,
    lambdaRLS,
    nTaps,
    storeCoeff,
    runWL,
    alg,
    constSymb,
    prec,
    blockLen=1,
):
    """
    Adaptive equalizer core processing function (block-wise)

    Parameters
    ----------
    sigIn : np.array
        Input signal array.
    symbRef : np.array
        Reference symbol sequence.
    SpS : int
        Samples per symbol.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    L : int
        Length of the output.
    mu : float
        Step size parameter.
    lambdaRLS : float
        RLS forgetting factor.
    nTaps : int
        Number of taps.
    storeCoeff : bool
        Flag indicating whether to store coefficient matrices.
    runWL : bool
        Run widely-linear mode
    alg : str
        Equalizer algorithm.
    constSymb : np.array
        Constellation symbols.
    prec : data type
        Precision of the computations.
    blockLen : int
        Length of the processing block.

    Returns
    -------
    sigOut : np.array
        Equalized output array.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    errSq : np.array
        Squared absolute error array.
    Hiter : np.array
        History of coefficient matrices.
    """
    nModes = sigIn.shape[1]
    indTaps = np.arange(0, nTaps)
    indMode = np.arange(0, nModes)

    errSq = np.zeros((nModes, L))
    sigIn = sigIn.astype(prec)
    H = H.astype(prec)
    H_ = H_.astype(prec)

    sigOut = np.zeros((L, nModes), dtype=prec)

    if storeCoeff:
        Hiter = np.zeros((nModes**2, nTaps, L), dtype=prec)
    else:
        Hiter = np.zeros((nModes**2, nTaps, 1), dtype=prec)

    Sd = np.eye(nTaps, dtype=prec)
    aux = Sd.copy()
    for _ in range(nTaps - 1):
        Sd = np.concatenate((Sd, aux))

    # Raios cma, rde
    Rcma = (
        np.mean(np.abs(constSymb) ** 4) / np.mean(np.abs(constSymb) ** 2)
    ) * np.ones((1, nModes)).astype(prec)
    Rrde = np.unique(np.abs(constSymb)).astype(prec)

    nBlocks = int(np.ceil(L / blockLen))

    for b in range(nBlocks):
        bStart = b * blockLen
        bEnd = min(bStart + blockLen, L)
        Lb = bEnd - bStart

        # --- coeficientes congelados no início do bloco ---
        Hb = H.copy()
        Hb_ = H_.copy()

        indInBlock = np.zeros((Lb, nTaps), dtype=np.int64)
        for k in range(Lb):
            ind = bStart + k
            for t in range(nTaps):
                indInBlock[k, t] = indTaps[t] + ind * SpS

        sigInBlock = np.zeros((Lb, nTaps, nModes), dtype=prec)
        for k in range(Lb):
            for t in range(nTaps):
                idx = indInBlock[k, t]
                for m in range(nModes):
                    sigInBlock[k, t, m] = sigIn[idx, m]

        # compute equalizer output for the block
        outBlock = np.zeros((Lb, nModes), dtype=prec)
        for N in range(nModes):
            inEqBlock = sigInBlock[:, :, N]  # (Lb, nTaps)
            outBlock += inEqBlock @ Hb[indMode + N * nModes, :].T
            if runWL:
                outBlock += inEqBlock.conjugate() @ Hb_[indMode + N * nModes, :].T

        for k in range(Lb):
            for m in range(nModes):
                sigOut[bStart + k, m] = outBlock[k, m]

        # update equalizer taps according to the specified algorithm and save squared error
        symbRefBlock = symbRef[bStart:bEnd, :]

        if alg == "nlms":
            H, H_, errBlock = nlmsUpBlock(
                sigInBlock, symbRefBlock, outBlock, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "cma":
            H, H_, errBlock = cmaUpBlock(
                sigInBlock, Rcma, outBlock, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "dd-lms":
            H, H_, errBlock = ddlmsUpBlock(
                sigInBlock, constSymb, outBlock, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "rde":
            H, H_, errBlock = rdeUpBlock(
                sigInBlock, Rrde, outBlock, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "da-rde":
            H, H_, errBlock = dardeUpBlock(
                sigInBlock, symbRefBlock, outBlock, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "rls":
            H, Sd, errBlock = rlsUpBlock(
                sigInBlock, symbRefBlock, outBlock, lambdaRLS, H, Sd, nModes, prec
            )
        elif alg == "dd-rls":
            H, Sd, errBlock = ddrlsUpBlock(
                sigInBlock, constSymb, outBlock, lambdaRLS, H, Sd, nModes, prec
            )
        elif alg == "static":
            errTemplate = np.abs(outBlock) ** 2
            errTemplate[:, :] = 0.0
            if bStart > 0:
                for m in range(nModes):
                    prevVal = errSq[m, bStart - 1]
                    for k in range(Lb):
                        errTemplate[k, m] = prevVal
            errBlock = errTemplate.T
        else:
            raise ValueError(
                "Equalization algorithm not specified (or incorrectly specified)."
            )

        for m in range(nModes):
            for k in range(Lb):
                errSq[m, bStart + k] = errBlock[m, k]

        if storeCoeff:
            for k in range(Lb):
                Hiter[:, :, bStart + k] = H
        else:
            Hiter[:, :, 0] = H

    return sigOut, H, H_, errSq, Hiter


@njit(fastmath=True)
def nlmsUpBlock(sigInBlock, symbRefBlock, outEqBlock, mu, H, H_, nModes, runWL, prec):
    """
    NLMS coefficient update for block processing.

    Parameters
    ----------
    sigInBlock : np.array
        Input signal block.
    symbRefBlock : np.array
        Reference symbol block.
    outEqBlock : np.array
        Equalized output block.
    mu : float
        Step size for the update.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec: data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    errSq : np.array
        Squared absolute error for the block.
    """
    Lb = sigInBlock.shape[0]
    nTaps = H.shape[1]
    indMode = np.arange(0, nModes)

    err = symbRefBlock - outEqBlock  # (Lb, nModes)

    for N in range(nModes):
        indUpdTaps = indMode + N * nModes
        inBlockN = sigInBlock[:, :, N]  # (Lb, nTaps)

        # normalização NLMS amostra a amostra (igual à versão original)
        normPerSample = np.sum(np.abs(inBlockN) ** 2, axis=1)  # (Lb,)
        inAdaptNorm = (inBlockN.T / normPerSample).T.astype(prec)  # (Lb, nTaps)

        grad = err.T.astype(prec) @ inAdaptNorm.conjugate()  # (nModes, nTaps)
        H[indUpdTaps, :] += mu * grad
        if runWL:
            gradWL = err.T.astype(prec) @ inAdaptNorm
            H_[indUpdTaps, :] += mu * gradWL

    return H, H_, (np.abs(err) ** 2).T


@njit(fastmath=True)
def ddlmsUpBlock(sigInBlock, constSymb, outEqBlock, mu, H, H_, nModes, runWL, prec):
    """
    DD-LMS coefficient update for block processing.

    Parameters
    ----------
    sigInBlock : np.array
        Input signal block.
    constSymb : np.array
        Array of constellation symbols.
    outEqBlock : np.array
        Equalized output block.
    mu : float
        Step size for the update.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec: data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    errSq : np.array
        Squared absolute error for the block.
    """
    Lb = sigInBlock.shape[0]
    indMode = np.arange(0, nModes)

    decided = np.zeros((Lb, nModes), dtype=prec)
    for n in range(Lb):
        for k in range(nModes):
            indSymb = np.argmin(np.abs(outEqBlock[n, k] - constSymb))
            decided[n, k] = constSymb[indSymb]
    err = (decided - outEqBlock).astype(prec)

    for N in range(nModes):
        indUpdTaps = indMode + N * nModes
        inBlockN = sigInBlock[:, :, N].astype(prec)

        grad = err.T @ inBlockN.conjugate()
        H[indUpdTaps, :] += mu * grad
        if runWL:
            H_[indUpdTaps, :] += mu * (err.T @ inBlockN)

    return H, H_, (np.abs(err) ** 2).T


@njit(fastmath=True)
def cmaUpBlock(sigInBlock, R, outEqBlock, mu, H, H_, nModes, runWL, prec):
    """
    CMA coefficient update for block processing.

    Parameters
    ----------
    sigInBlock : np.array
        Input signal block.
    R : np.array
        Correlation array.
    outEqBlock : np.array
        Equalized output block.
    mu : float
        Step size for the update.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec: data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    errSq : np.array
        Squared absolute error for the block.
    """
    indMode = np.arange(0, nModes)

    err = (R - np.abs(outEqBlock) ** 2).astype(prec)  # (Lb, nModes), R faz broadcast
    prodErrOut = err * outEqBlock.astype(prec)  # equivalente a diag(err) @ diag(outEq)

    for N in range(nModes):
        indUpdTaps = indMode + N * nModes
        inBlockN = sigInBlock[:, :, N].astype(prec)

        grad = prodErrOut.T @ inBlockN.conjugate()
        H[indUpdTaps, :] += mu * grad
        if runWL:
            H_[indUpdTaps, :] += mu * (prodErrOut.T @ inBlockN)

    return H, H_, (np.abs(err) ** 2).T


@njit(fastmath=True)
def rdeUpBlock(sigInBlock, R, outEqBlock, mu, H, H_, nModes, runWL, prec):
    """
    RDE coefficient update for block processing.

    Parameters
    ----------
    sigInBlock : np.array
        Input signal block.
    R : np.array
        Constellation radius array.
    outEqBlock : np.array
        Equalized output block.
    mu : float
        Step size for the update.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec: data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    errSq : np.array
        Squared absolute error for the block.
    """
    Lb = sigInBlock.shape[0]
    indMode = np.arange(0, nModes)

    decidedR = np.zeros((Lb, nModes), dtype=prec)
    for n in range(Lb):
        for k in range(nModes):
            indR = np.argmin(np.abs(R - np.abs(outEqBlock[n, k])))
            decidedR[n, k] = R[indR]

    err = (decidedR**2 - np.abs(outEqBlock) ** 2).astype(prec)
    prodErrOut = err * outEqBlock.astype(prec)

    for N in range(nModes):
        indUpdTaps = indMode + N * nModes
        inBlockN = sigInBlock[:, :, N].astype(prec)

        grad = prodErrOut.T @ inBlockN.conjugate()
        H[indUpdTaps, :] += mu * grad
        if runWL:
            H_[indUpdTaps, :] += mu * (prodErrOut.T @ inBlockN)

    return H, H_, (np.abs(err) ** 2).T


@njit(fastmath=True)
def dardeUpBlock(sigInBlock, refBlock, outEqBlock, mu, H, H_, nModes, runWL, prec):
    """
    Data-aided RDE coefficient update for block processing.

    Parameters
    ----------
    sigInBlock : np.array
        Input signal block.
    refBlock : np.array
        Reference symbol block.
    outEqBlock : np.array
        Equalized output block.
    mu : float
        Step size for the update.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec: data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    errSq : np.array
        Squared absolute error for the block.
    """
    indMode = np.arange(0, nModes)

    decidedR = np.abs(refBlock).astype(prec)  # (Lb, nModes)
    err = (decidedR**2 - np.abs(outEqBlock) ** 2).astype(prec)
    prodErrOut = err * outEqBlock.astype(prec)

    for N in range(nModes):
        indUpdTaps = indMode + N * nModes
        inBlockN = sigInBlock[:, :, N].astype(prec)

        grad = prodErrOut.T @ inBlockN.conjugate()
        H[indUpdTaps, :] += mu * grad
        if runWL:
            H_[indUpdTaps, :] += mu * (prodErrOut.T @ inBlockN)

    return H, H_, (np.abs(err) ** 2).T


@njit(fastmath=True)
def rlsUpBlock(sigInBlock, symbRefBlock, outEqBlock, λ, H, Sd, nModes, prec):
    """
    RLS coefficient update for block processing.

    Parameters
    ----------
    sigInBlock : np.array
        Input signal block.
    symbRefBlock : np.array
        Reference symbol block.
    outEqBlock : np.array
        Equalized output block.
    λ : float
        Forgetting factor for the RLS algorithm.
    H : np.array
        Coefficient matrix.
    Sd : np.array
        Inverse correlation matrix.
    nModes : int
        Number of modes.
    prec: data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    Sd : np.array
        Updated inverse correlation matrix.
    errSq : np.array
        Squared absolute error for the block.

    """
    Lb = sigInBlock.shape[0]
    nTaps = H.shape[1]
    indMode = np.arange(0, nModes)
    indTaps = np.arange(0, nTaps)
    Sd = Sd.astype(prec)

    err = (symbRefBlock - outEqBlock).astype(prec)  # (Lb, nModes), já com H congelado

    for N in range(nModes):
        indUpdModes = indMode + N * nModes
        indUpdTaps = indTaps + N * nTaps

        Sd_ = Sd[indUpdTaps, :]
        deltaHN = np.zeros((nModes, nTaps), dtype=prec)

        for n in range(Lb):
            inAdapt = sigInBlock[n, :, N].conjugate().reshape(-1, 1).astype(prec)
            inAdaptPar = ((inAdapt.T).repeat(nModes).reshape(nTaps, -1).T).astype(prec)

            A = (Sd_ @ inAdapt).astype(prec)
            B = (inAdapt.conjugate().astype(prec).T @ Sd_).astype(prec)
            C = (inAdapt.conjugate().astype(prec).T @ A).astype(prec)
            num = (A @ B).astype(prec)

            Sd_ = ((1 / λ) * (Sd_ - num / (λ + C))).astype(prec)

            Y = (Sd_ @ inAdaptPar.T).astype(prec).T  # (nModes, nTaps)
            errDiag_n = np.diag(err[n, :]).astype(prec)
            deltaHN += errDiag_n @ Y

        H[indUpdModes, :] += deltaHN
        Sd[indUpdTaps, :] = Sd_

    return H, Sd, (np.abs(err) ** 2).T


@njit(fastmath=True)
def ddrlsUpBlock(sigInBlock, constSymb, outEqBlock, λ, H, Sd, nModes, prec):
    """
    DD-RLS coefficient update for block processing.

    Parameters
    ----------
    sigInBlock : np.array
        Input signal block.
    constSymb : np.array
        Array of constellation symbols.
    outEqBlock : np.array
        Equalized output block.
    λ : float
        Forgetting factor for the RLS algorithm.
    H : np.array
        Coefficient matrix.
    Sd : np.array
        Inverse correlation matrix.
    nModes : int
        Number of modes.
    prec: data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    Sd : np.array
        Updated inverse correlation matrix.
    errSq : np.array
        Squared absolute error for the block.
    """
    Lb = sigInBlock.shape[0]
    nTaps = H.shape[1]
    indMode = np.arange(0, nModes)
    indTaps = np.arange(0, nTaps)
    Sd = Sd.astype(prec)

    decided = np.zeros((Lb, nModes), dtype=prec)
    for n in range(Lb):
        for k in range(nModes):
            indSymb = np.argmin(np.abs(outEqBlock[n, k] - constSymb))
            decided[n, k] = constSymb[indSymb]
    err = (decided - outEqBlock).astype(prec)  # (Lb, nModes)

    for N in range(nModes):
        indUpdModes = indMode + N * nModes
        indUpdTaps = indTaps + N * nTaps

        Sd_ = Sd[indUpdTaps, :]
        deltaHN = np.zeros((nModes, nTaps), dtype=prec)

        for n in range(Lb):
            inAdapt = sigInBlock[n, :, N].conjugate().reshape(-1, 1).astype(prec)
            inAdaptPar = ((inAdapt.T).repeat(nModes).reshape(nTaps, -1).T).astype(prec)

            A = (Sd_ @ inAdapt).astype(prec)
            B = (inAdapt.conjugate().astype(prec).T @ Sd_).astype(prec)
            C = (inAdapt.conjugate().astype(prec).T @ A).astype(prec)
            num = (A @ B).astype(prec)

            Sd_ = ((1 / λ) * (Sd_ - num / (λ + C))).astype(prec)

            Y = (Sd_ @ inAdaptPar.T).astype(prec).T
            errDiag_n = np.diag(err[n, :]).astype(prec)
            deltaHN += errDiag_n @ Y

        H[indUpdModes, :] += deltaHN
        Sd[indUpdTaps, :] = Sd_

    return H, Sd, (np.abs(err) ** 2).T


@njit(fastmath=True)
def realValuedDFECore(
    sigIn,
    symbRef,
    nTapsFF=5,
    nTapsFB=5,
    SpS=1,
    mu=0.0001,
    nTrain=1000,
    prec=np.float32,
    constSymb=None,
    f=None,
    b=None,
    trainingMode="data-aided",
    preconvIters=1,
):
    """
    Decision feedback equalizer (DFE) core implementation.

    Parameters
    ----------
    sigIn : np.array
        Input signal to be equalized.
    symbRef : np.array
        Desired (reference) signal.
    nTapsFF : int
        Number of feedforward taps
    nTapsFB : int
        Number of feedback taps
    SpS : int
        Samples per symbol
    mu : float
        Step size
    nTrain : int
        Number of training symbols
    prec : data type
        Precision
    constSymb : np.array
        Array of constellation symbols used for symbol decisions.
    f : np.array
        Initial feedforward coeffs
    b : np.array
        Initial feedback coeffs
    trainingMode : str
        Operation mode ('data-aided', 'fulltime')
    preconvIters : int
        Number of pre-convergence iterations

    Returns
    -------
    sigOut : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.
    b : np.array
        Final feedback filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.

    """
    L = len(sigIn)  # number of input samples
    N = int((L - nTapsFF + nTapsFF % 2) // SpS)  # number of input symbols

    # Buffers
    xbuf = sigIn[0:nTapsFF].astype(prec)  # past input samples
    dbuf = np.zeros(nTapsFB, dtype=prec)  # past decisions
    sigOut = np.zeros(N, dtype=prec)
    mse = np.zeros(N, dtype=prec)

    nIter = 1
    k = 0
    while k < N:
        # Compute output
        sigOut[k] = np.dot(f, xbuf) + np.dot(b, dbuf)

        # Reference for adaptation: training then decision-directed
        if k < nTrain:
            outRef = symbRef[k]
        else:
            indSymb = np.argmin(np.abs(sigOut[k] - constSymb))
            outRef = constSymb[indSymb]

        # Error
        ek = outRef - sigOut[k]
        mse[k] = ek**2

        if (trainingMode == "data-aided" and k < nTrain) or (
            trainingMode == "fulltime"
        ):
            # LMS updates
            f += mu * ek * xbuf
            b += mu * ek * dbuf

        # Update feedback buffer with the new decision
        if nTapsFB > 0:
            dbuf = np.roll(dbuf, 1)
            dbuf[0] = outRef

        # Update FF buffer:
        xbuf = np.roll(xbuf, -SpS)
        firstSample = int(k * SpS + nTapsFF)
        lastSample = int(firstSample + SpS)

        # Fill the last SpS samples
        if lastSample < L:
            for i in range(SpS):
                xbuf[-SpS + i] = sigIn[firstSample + i]
        else:
            for i in range(SpS):
                xbuf[-SpS + i] = 0.0

        if k == nTrain and nIter < preconvIters:
            k = 0  # restart pre-convergence
            nIter += 1
        else:
            k += 1

    return sigOut, f, b, mse


@njit(fastmath=True)
def complexValuedDFECore(
    sigIn,
    symbRef,
    nTapsFF=5,
    nTapsFB=5,
    SpS=1,
    mu=0.0001,
    nTrain=1000,
    prec=np.complex64,
    constSymb=None,
    f=None,
    b=None,
    trainingMode="data-aided",
    preconvIters=1,
):
    """
    Decision feedback equalizer (DFE) core implementation for complex-valued signals.

    Parameters
    ----------
    sigIn : np.array
        Input signal to be equalized.
    symbRef : np.array
        Desired (reference) signal.
    nTapsFF : int
        Number of feedforward taps
    nTapsFB : int
        Number of feedback taps
    SpS : int
        Samples per symbol
    mu : float
        Step size
    nTrain : int
        Number of training symbols
    prec : data type
        Precision
    constSymb : np.array
        Constellation symbols
    f : np.array
        Initial feedforward filter coefficients.
    b : np.array
        Initial feedback filter coefficients.
    trainingMode : str
        Operation mode ('data-aided', 'fulltime')
    preconvIters : int
        Number of pre-convergence iterations

    Returns
    -------
    sigOut : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.
    b : np.array
        Final feedback filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.

    """
    L = len(sigIn)  # number of input samples
    N = int((L - nTapsFF + nTapsFF % 2) // SpS)  # number of input symbols

    # Buffers
    xbuf = sigIn[0:nTapsFF].astype(prec)  # past input samples
    dbuf = np.zeros(nTapsFB, dtype=prec)  # past decisions
    sigOut = np.zeros(N, dtype=prec)
    mse = np.zeros(N, dtype=prec)

    nIter = 1
    k = 0
    while k < N:
        # Compute output
        sigOut[k] = np.dot(f, xbuf) + np.dot(b, dbuf)

        # Reference for adaptation: training then decision-directed
        if k < nTrain:
            outRef = symbRef[k]
        else:
            indSymb = np.argmin(np.abs(sigOut[k] - constSymb))
            outRef = constSymb[indSymb]

        # Error
        ek = outRef - sigOut[k]
        mse[k] = np.abs(ek) ** 2

        if (trainingMode == "data-aided" and k < nTrain) or (
            trainingMode == "fulltime"
        ):
            # LMS updates
            f += mu * ek * xbuf.conjugate()
            b += mu * ek * dbuf.conjugate()

        # Update feedback buffer with the new decision
        if nTapsFB > 0:
            dbuf = np.roll(dbuf, 1)
            dbuf[0] = outRef

        # Update FF buffer:
        xbuf = np.roll(xbuf, -SpS)
        firstSample = int(k * SpS + nTapsFF)
        lastSample = int(firstSample + SpS)

        # Fill the last SpS samples
        if lastSample < L:
            for i in range(SpS):
                xbuf[-SpS + i] = sigIn[firstSample + i]
        else:
            for i in range(SpS):
                xbuf[-SpS + i] = 0.0 + 0.0 * 1j

        if k == nTrain and nIter < preconvIters:
            k = 0  # restart pre-convergence
            nIter += 1
        else:
            k += 1

    return sigOut, f, b, mse


@njit(fastmath=True)
def realValuedFFECore(
    sigIn,
    symbRef,
    nTaps=5,
    SpS=1,
    mu=0.0001,
    nTrain=1000,
    prec=np.float32,
    constSymb=None,
    f=None,
    trainingMode="data-aided",
    preconvIters=1,
):
    """
    Decision-directed feedforward equalizer (FFE) core implementation.

    Parameters
    ----------
    sigIn : np.array
        Input signal to be equalized.
    symbRef : np.array
        Desired (reference) signal.
    nTaps : int
        Number of feedforward taps
    SpS : int
        Samples per symbol
    mu : float
        Step size
    nTrain : int
        Number of training symbols
    prec : data type
        Precision
    constSymb : np.array
        Constellation symbols
    f : np.array
        Initial feedforward filter coefficients
    trainingMode : str
        Operation mode ('data-aided', 'fulltime')
    preconvIters : int
        Number of pre-convergence iterations

    Returns
    -------
    sigOut : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.

    """
    L = len(sigIn)  # number of input samples
    N = int((L - nTaps + nTaps % 2) // SpS)  # number of input symbols

    # Buffer
    sigOut = np.zeros(N, dtype=prec)
    mse = np.zeros(N, dtype=prec)
    xbuf = sigIn[0:nTaps].astype(prec)  # past input samples

    nIter = 1
    k = 0
    while k < N:
        # Compute output
        sigOut[k] = np.dot(f, xbuf)

        # Reference for adaptation: training then decision-directed
        if k < nTrain:
            outRef = symbRef[k]
        else:
            indSymb = np.argmin(np.abs(sigOut[k] - constSymb))
            outRef = constSymb[indSymb]

        # Error
        ek = outRef - sigOut[k]
        mse[k] = ek**2

        if (trainingMode == "data-aided" and k < nTrain) or (
            trainingMode == "fulltime"
        ):
            # LMS update
            f += mu * ek * xbuf

        # Update FF buffer:
        xbuf = np.roll(xbuf, -SpS)
        firstSample = int(k * SpS + nTaps)
        lastSample = int(firstSample + SpS)

        # Fill the last SpS samples
        if lastSample < L:
            for i in range(SpS):
                xbuf[-SpS + i] = sigIn[firstSample + i]
        else:
            for i in range(SpS):
                xbuf[-SpS + i] = 0.0

        if k == nTrain and nIter < preconvIters:
            k = 0  # restart pre-convergence
            nIter += 1
        else:
            k += 1

    return sigOut, f, mse


@njit(fastmath=True)
def complexValuedFFECore(
    sigIn,
    symbRef,
    nTaps=5,
    SpS=1,
    mu=0.0001,
    nTrain=1000,
    prec=np.complex64,
    constSymb=None,
    f=None,
    trainingMode="data-aided",
    preconvIters=1,
):
    """
    Decision-directed feedforward equalizer (FFE) core implementation for complex-valued signals.

    Parameters
    ----------
    sigIn : np.array
        Input signal to be equalized.
    symbRef : np.array
        Desired (reference) signal.
    nTaps : int
        Number of feedforward taps
    SpS : int
        Samples per symbol
    mu : float
        Step size
    nTrain : int
        Number of training symbols
    prec : data type
        Precision
    constSymb : np.array
        Constellation symbols
    f : np.array
        Initial feedforward filter coefficients
    trainingMode : str
        Operation mode ('data-aided', 'fulltime')
    preconvIters : int
        Number of pre-convergence iterations

    Returns
    -------
    sigOut : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.
    """
    L = len(sigIn)  # number of input samples
    N = int((L - nTaps + nTaps % 2) // SpS)  # number of input symbols

    # Buffer
    sigOut = np.zeros(N, dtype=prec)
    mse = np.zeros(N, dtype=prec)
    xbuf = sigIn[0:nTaps].astype(prec)  # past input samples

    nIter = 1
    k = 0
    while k < N:
        # Compute output
        sigOut[k] = np.dot(f, xbuf)

        # Reference for adaptation: training then decision-directed
        if k < nTrain:
            outRef = symbRef[k]
        else:
            indSymb = np.argmin(np.abs(sigOut[k] - constSymb))
            outRef = constSymb[indSymb]

        # Error
        ek = outRef - sigOut[k]
        mse[k] = np.abs(ek) ** 2

        if (trainingMode == "data-aided" and k < nTrain) or (
            trainingMode == "fulltime"
        ):
            # LMS update
            f += mu * ek * xbuf.conjugate()

        # Update FF buffer:
        xbuf = np.roll(xbuf, -SpS)
        firstSample = int(k * SpS + nTaps)
        lastSample = int(firstSample + SpS)

        # Fill the last SpS samples
        if lastSample < L:
            for i in range(SpS):
                xbuf[-SpS + i] = sigIn[firstSample + i]
        else:
            for i in range(SpS):
                xbuf[-SpS + i] = 0.0 + 0.0 * 1j

        if k == nTrain and nIter < preconvIters:
            k = 0  # restart pre-convergence
            nIter += 1
        else:
            k += 1

    return sigOut, f, mse


@njit(fastmath=True)
def volterraCore(
    sigIn,
    symbRef,
    order=2,
    SpS=1,
    mu=0.0001,
    nTrain=1000,
    h1=None,
    h2=None,
    h3=None,
    prec=np.float32,
    constSymb=None,
    trainingMode="data-aided",
    preconvIters=1,
):
    """
    Decision-directed Volterra equalizer core implementation

    Parameters
    ----------
    sigIn : np.array
        Input signal to be equalized.
    symbRef : np.array
        Desired (reference) signal.
    order : int
        Volterra series order (2 for quadratic, 3 for cubic)
    SpS : int
        Samples per symbol
    mu : float
        Step size
    nTrain : int
        Number of training symbols
    h1 : np.array
        Initial linear filter coefficients.
    h2 : np.array
        Initial quadratic filter coefficients.
    h3 : np.array
        Initial cubic filter coefficients.
    prec : data type
        Precision
    constSymb : np.array
        Constellation symbols
    trainingMode : str
        Operation mode ('data-aided', 'fulltime')
    preconvIters : int
        Number of pre-convergence iterations

    Returns
    -------
    sigOut : np.array
        Equalized output signal.
    h1 : np.array
        Final linear filter coefficients.
    h2 : np.array
        Final quadratic filter coefficients.
    h3 : np.array
        Final cubic filter coefficients.

    References
    ----------
    [1] Diniz, P. R., da Silva, E. A. B., & Netto, S. L. Adaptive Filtering: Algorithms and Practical Implementation. Springer Science & Business Media, 2010.

    """
    n1Taps = h1.shape[0]
    n2Taps = h2.shape[0]
    n3Taps = h3.shape[0]

    nTaps = np.max(np.array([n1Taps, n2Taps, n3Taps]))
    L = len(sigIn)  # number of input samples
    N = int((L - nTaps + nTaps % 2) // SpS)  # number of input symbols

    # initialize outputs
    sigOut = np.zeros(N, dtype=prec)
    mse = np.zeros(N, dtype=prec)

    t2 = int((n1Taps - n2Taps) // 2)
    t3 = int((n1Taps - n3Taps) // 2)

    # Buffer
    xbuf = sigIn[0:nTaps].astype(prec)  # past input samples

    nIter = 1
    k = 0
    while k < N:
        # Compute output
        linearPart = np.dot(h1, xbuf)
        quadraticPart = 0.0
        cubicPart = 0.0

        for i in range(n2Taps):
            for j in range(n2Taps):
                quadraticPart += h2[i, j] * xbuf[t2 + i] * xbuf[t2 + j]

        if order == 3:
            for i in range(n3Taps):
                for j in range(n3Taps):
                    for l in range(n3Taps):
                        cubicPart += (
                            h3[i, j, l] * xbuf[t3 + i] * xbuf[t3 + j] * xbuf[t3 + l]
                        )

        sigOut[k] = linearPart + quadraticPart + cubicPart

        # Reference for adaptation: training then decision-directed
        if k < nTrain:
            outRef = symbRef[k]
        else:
            indSymb = np.argmin(np.abs(sigOut[k] - constSymb))
            outRef = constSymb[indSymb]

        # Error
        ek = outRef - sigOut[k]
        mse[k] = ek**2

        if (trainingMode == "data-aided" and k < nTrain) or (
            trainingMode == "fulltime"
        ):

            # LMS updates

            # Update linear coefficients
            h1 += mu * ek * xbuf

            # Update quadratic coefficients
            for i in range(n2Taps):
                for j in range(n2Taps):
                    h2[i, j] += mu / 2 * ek * xbuf[t2 + i] * xbuf[t2 + j]

            # Update cubic coefficients
            if order == 3:
                for i in range(n3Taps):
                    for j in range(n3Taps):
                        for l in range(n3Taps):
                            h3[i, j, l] += (
                                mu / 7 * ek * xbuf[t3 + i] * xbuf[t3 + j] * xbuf[t3 + l]
                            )

        # Update FF buffer:
        xbuf = np.roll(xbuf, -SpS)
        firstSample = int(k * SpS + nTaps)
        lastSample = int(firstSample + SpS)

        # Fill the last SpS samples
        if lastSample < L:
            for i in range(SpS):
                xbuf[-SpS + i] = sigIn[firstSample + i]
        else:
            for i in range(SpS):
                xbuf[-SpS + i] = 0.0

        if k == nTrain and nIter < preconvIters:
            k = 0  # restart pre-convergence
            nIter += 1
        else:
            k += 1

    return sigOut, h1, h2, h3, mse


def coreAdaptEqFD(
    sigIn,
    symbRef,
    SpS,
    H,
    L,
    mu,
    nTaps,
    Nfft,
    storeCoeff,
    alg,
    constSymb,
    prec=np.complex128,
):
    """
    Núcleo do equalizador adaptativo MIMO N x N no domínio da frequência,
    usando o método overlap-and-save (generalização do `fdlms` fornecido
    como exemplo para o caso 1x1/LMS).

    Diferenças estruturais em relação a `coreAdaptEq` (domínio do tempo):

    - Os coeficientes H mantêm o mesmo layout 2D de `coreAdaptEq`
      (linha = indMode + N*nModes, uma linha por par modo de saída /
      modo de entrada), só que cada linha agora tem comprimento Nfft
      (resposta em frequência) em vez de nTaps (resposta ao impulso).
    - O sinal é processado em blocos de Nfft amostras (overlap-and-save),
      e não amostra a amostra.
    - O downsampling de SpS -> 1 amostra/símbolo é feito NO DOMÍNIO DA
      FREQUÊNCIA, antes da IFFT: aplica-se um filtro brickwall
      anti-aliasing (mantém só a banda base |f| < Nyquist/SpS) e então
      dizima-se o espectro (soma das réplicas/"folding", técnica padrão
      de downsampling em frequência), de modo que o bloco decimado é
      obtido com uma única IFFT de tamanho Nfft/SpS, sem nunca passar
      pela versão à taxa de amostra plena no domínio do tempo.
    - Como o equalizador é fracionalmente espaçado (entrada a SpS
      amostras/símbolo, saída a 1 amostra/símbolo), o erro usado no
      gradiente é "zero-stuffed" na taxa de amostra antes da FFT --
      generalização direta do `fft(error)*X.conj()` do `fdlms` (que
      assume SpS = 1) para o caso decimado. Essa etapa continua no
      domínio da frequência plena (Nfft), já que ali X também está na
      taxa de amostra plena.
    - RLS / dd-rls não têm forma de bloco/FFT direta (dependem de
      atualização recursiva amostra a amostra com lema de inversão de
      matriz) e não são suportados aqui.
    - O modo widely-linear (H_) não foi implementado nesta versão; veja
      a nota no final do arquivo para como estendê-lo.

    Parameters
    ----------
    sigIn : np.array, shape (Nin, nModes)
        Sinal de entrada (amostrado a SpS amostras por símbolo).
    symbRef : np.array, shape (L, nModes)
        Sequência de símbolos de referência (necessária para 'nlms' e
        'da-rde'; ignorada nos algoritmos cegos).
    SpS : int
        Amostras por símbolo.
    H : np.array, shape (nModes**2, Nfft) ou None
        Coeficientes iniciais no domínio da frequência, mesmo layout de
        `coreAdaptEq`: a linha H[indMode + N*nModes, :] é a resposta em
        frequência aplicada ao modo de entrada N que contribui para a
        saída do modo indMode (aqui com Nfft amostras em vez de nTaps).
        Se None, inicia em zero.
    L : int
        Número de símbolos de saída desejados.
    mu : float
        Passo de adaptação.
    nTaps : int
        Comprimento efetivo do filtro no domínio do tempo (equivalente).
    Nfft : int
        Tamanho da FFT (será ajustado para o menor valor >= Nfft da
        forma SpS * 2**m, garantindo que seja múltiplo de SpS e que o
        bloco decimado Nfft/SpS seja potência de 2).
    storeCoeff : bool
        Se True, guarda o histórico de H a cada bloco.
    alg : str
        Um de {'nlms', 'cma', 'dd-lms', 'rde', 'da-rde', 'static'}.
    constSymb : np.array
        Símbolos da constelação.
    prec : dtype
        Precisão dos cálculos [default: np.complex128].

    Returns
    -------
    sigOut : np.array, shape (L, nModes)
        Sinal equalizado (um símbolo por linha).
    H : np.array, shape (nModes**2, Nfft)
        Coeficientes finais no domínio da frequência.
    errSq : np.array, shape (nModes, L)
        Erro quadrático absoluto por símbolo.
    Hiter : np.array, shape (nModes**2, Nfft, nblocks ou 1)
        Histórico de H por bloco (se storeCoeff) ou apenas o valor final.
    """
    nModes = int(sigIn.shape[1])
    indMode = np.arange(nModes)
    sigIn = np.asarray(sigIn).astype(prec)
    symbRef = np.asarray(symbRef).astype(prec)
    constSymb = np.asarray(constSymb).astype(prec)

    # --- parâmetros de bloco / overlap-and-save -------------------------
    # Nfft ajustado para SpS * 2**m: garante Nfft múltiplo de SpS (para o
    # downsampling em frequência) e Nfft/SpS potência de 2 (para a IFFT
    # decimada).
    m = int(np.ceil(np.log2(max(Nfft / SpS, 1))))
    Nfft = SpS * (2**m)
    if Nfft <= nTaps:
        raise ValueError("Nfft deve ser maior que nTaps.")

    Md = Nfft // SpS  # tamanho do bloco após decimação em frequência
    bFull = Nfft - nTaps + 1  # amostras válidas (taxa de amostra) por bloco
    nSymb = bFull // SpS  # símbolos de saída gerados por bloco
    if nSymb < 1:
        raise ValueError(
            "Combinação de Nfft, nTaps e SpS inválida: nenhum símbolo de "
            "saída cabe em um bloco. Aumente Nfft."
        )
    if nSymb > Md:
        raise ValueError(
            "nSymb > Nfft/SpS: aumente Nfft (não deveria ocorrer em "
            "condições normais)."
        )
    bAdv = nSymb * SpS  # avanço do bloco em amostras (múltiplo de SpS)
    indTapsOffset = nTaps - 1  # amostras iniciais descartadas (wraparound circular)

    # --- filtro brickwall anti-aliasing + deslocamento de fase -----------
    # Máscara brickwall: mantém só os bins de frequência dentro da banda
    # base decimada, |f| < Nyquist/SpS (evita aliasing no downsampling).
    half = Md // 2
    brickwall = np.zeros(Nfft, dtype=prec)
    brickwall[:half] = 1.0
    brickwall[Nfft - half :] = 1.0

    # Deslocamento circular de +indTapsOffset amostras (avanço, não atraso;
    # implementado como multiplicação por rampa de fase no domínio da
    # frequência: z[n] = y[(n+n0) mod Nfft]  <=>  Z[k] = Y[k]*e^{+j2πk n0/Nfft}),
    # para que a decimação comece exatamente na primeira amostra válida do
    # bloco (mesmo papel do slice yBlock[indTapsOffset:...] da versão anterior).
    phaseShift = np.exp(1j * 2 * np.pi * np.arange(Nfft) * indTapsOffset / Nfft).astype(
        prec
    )

    filtShift = (brickwall * phaseShift).astype(prec)  # aplicado por modo de saída

    nblocks = int(np.ceil(L / nSymb))
    Lsymb = nblocks * nSymb  # total de símbolos processados (com zero-padding no fim)

    # --- prepend de histórico (nTaps-1 zeros), igual ao fdlms -----------
    sigIn = np.concatenate((np.zeros((nTaps - 1, nModes), dtype=prec), sigIn), axis=0)
    nSampNeeded = (nblocks - 1) * bAdv + Nfft
    if sigIn.shape[0] < nSampNeeded:
        sigIn = np.concatenate(
            (sigIn, np.zeros((nSampNeeded - sigIn.shape[0], nModes), dtype=prec)),
            axis=0,
        )

    symbRefPad = np.concatenate(
        (symbRef, np.zeros((Lsymb - L, nModes), dtype=prec)), axis=0
    )

    # --- coeficientes no domínio da frequência (layout 2D, igual a
    # coreAdaptEq: linha = indMode + N*nModes) --------------------------
    if H is None:
        # Inicialização "center-spike" (tap central = 1 na diagonal, i.e.
        # H[i+i*nModes,:] ~ identidade): necessária para os algoritmos
        # cegos (cma/rde/da-rde), cujo erro é proporcional à própria
        # saída do equalizador -- com H = 0 a saída fica em 0 para
        # sempre e o gradiente nunca sai do zero. Equivalente ao que se
        # faz tipicamente em coreAdaptEq antes de chamar o núcleo.
        Htime = np.zeros((nModes**2, nTaps), dtype=prec)
        center = nTaps // 2
        for i in range(nModes):
            Htime[i + i * nModes, center] = 1.0
        H = fft(Htime, n=Nfft, axis=1).astype(prec)
    else:
        H = np.asarray(H).astype(prec).copy()
        if H.ndim != 2 or H.shape[0] != nModes**2:
            raise ValueError(
                f"H deve ter shape (nModes**2, Nfft) = ({nModes**2}, {Nfft}); "
                f"recebido {H.shape}."
            )
        if H.shape[1] != Nfft:
            raise ValueError(
                f"H deve ter {Nfft} colunas (Nfft), recebido {H.shape[1]}. "
                "Se H veio do domínio do tempo (nTaps colunas), faça o "
                "zero-padding para Nfft e a FFT antes de passar aqui."
            )

    sigOut = np.zeros((Lsymb, nModes), dtype=prec)
    errSq = np.zeros((nModes, Lsymb))

    if storeCoeff:
        Hiter = np.zeros((nModes**2, Nfft, nblocks), dtype=prec)
    else:
        Hiter = np.zeros((nModes**2, Nfft, 1), dtype=prec)

    # raios para cma/rde/da-rde, mesma definição do código original
    Rcma = (
        np.mean(np.abs(constSymb) ** 4) / np.mean(np.abs(constSymb) ** 2)
    ) * np.ones((1, nModes)).astype(prec)
    Rrde = np.unique(np.abs(constSymb)).astype(prec)

    if alg in ("rls", "dd-rls"):
        raise NotImplementedError(
            f"O algoritmo '{alg}' não possui uma forma de bloco/FFT direta "
            "(RLS depende de atualização recursiva amostra a amostra via "
            "lema de inversão de matriz). Use uma formulação de RLS no "
            "domínio da frequência dedicada, ou mantenha esse algoritmo "
            "no domínio do tempo."
        )

    for blk in range(nblocks):
        nStart = blk * bAdv

        # --- FFT do bloco de entrada, por modo ---------------------------
        X = np.zeros((Nfft, nModes), dtype=prec)
        for j in range(nModes):
            X[:, j] = fft(sigIn[nStart : nStart + Nfft, j])

        # --- filtragem: Y_i = soma_j H[i + j*nModes] * X_j -----------------
        # (mesma convenção de índice de coreAdaptEq: H[indMode + N*nModes,:])
        Yfd = np.zeros((Nfft, nModes), dtype=prec)
        for i in range(nModes):
            acc = np.zeros(Nfft, dtype=prec)
            for j in range(nModes):
                acc += H[i + j * nModes, :] * X[:, j]
            Yfd[:, i] = acc

        # --- downsampling de SpS -> 1 amostra/símbolo NO DOMÍNIO DA
        # FREQUÊNCIA, antes da IFFT: aplica o brickwall anti-aliasing +
        # deslocamento de fase (alinha com a região válida do overlap-
        # -save) e então dizima o espectro por "folding" (soma das SpS
        # réplicas de tamanho Md = Nfft/SpS); como o brickwall já
        # eliminou tudo fora da banda base decimada, essa soma reduz-se
        # exatamente ao espectro da versão decimada do sinal, sem
        # aliasing. Uma única IFFT de tamanho Md entrega o bloco já na
        # taxa de símbolo.
        Yproc = Yfd * filtShift[:, None]  # (Nfft, nModes)
        Ydec = Yproc.reshape(SpS, Md, nModes).sum(axis=0) / SpS  # (Md, nModes)
        outDecBlock = ifft(Ydec, axis=0)  # (Md, nModes), taxa de símbolo

        outBlock = outDecBlock[:nSymb, :]  # (nSymb, nModes)

        symbIdx0 = blk * nSymb
        refBlock = symbRefPad[symbIdx0 : symbIdx0 + nSymb, :]

        sigOut[symbIdx0 : symbIdx0 + nSymb, :] = outBlock

        # --- erro por símbolo, de acordo com o algoritmo ------------------
        errBlock, sqErrBlock = _fdErrorBlock(
            outBlock, refBlock, constSymb, Rcma, Rrde, alg
        )
        errSq[:, symbIdx0 : symbIdx0 + nSymb] = sqErrBlock.T

        # --- gradiente e atualização no domínio da frequência -------------
        # "zero-stuffing" do erro na taxa de amostra, alinhado com as
        # posições usadas na filtragem -- generalização de
        # `fft(error)*X.conj()` do fdlms para o caso SpS > 1.
        for i in range(nModes):
            errFull = np.zeros(Nfft, dtype=prec)
            errFull[indTapsOffset : indTapsOffset + bAdv : SpS] = errBlock[:, i]
            E = fft(errFull)
            for j in range(nModes):
                if alg == "nlms":
                    norm = 1  # np.mean(np.abs(X[:, j]) ** 2) + 1e-12
                    H[i + j * nModes, :] += (mu / norm) * E * np.conj(X[:, j])
                else:
                    H[i + j * nModes, :] += mu * E * np.conj(X[:, j]) / np.sqrt(Nfft)

        if storeCoeff:
            Hiter[:, :, blk] = H
        else:
            Hiter[:, :, 0] = H

    # descarta o zero-padding do final, devolve só os L símbolos pedidos
    sigOut = sigOut[:L, :]
    errSq = errSq[:, :L]

    return sigOut, H, errSq, Hiter


def _fdErrorBlock(outBlock, refBlock, constSymb, Rcma, Rrde, alg):
    """
    Erro por símbolo para um bloco (nSymb, nModes), de acordo com `alg`,
    seguindo exatamente as definições das funções *Up originais
    (nlmsUp, ddlmsUp, cmaUp, rdeUp, dardeUp).

    Duas grandezas são distinguidas, pois nos algoritmos cegos/RDE elas
    NÃO coincidem:

    - errRaw : o "err" das funções originais (usado para errSq / MSE).
    - errGrad: o termo que de fato multiplica conj(inAdapt) na
      atualização dos coeficientes (== errDiag @ ... ou
      prodErrOut = diag(err) @ diag(outEq), conforme o caso).

    Em 'nlms' e 'dd-lms', errGrad == errRaw (erro direto). Em 'cma',
    'rde' e 'da-rde', errGrad = errRaw * outBlock (produto erro-saída,
    como em `prodErrOut` nas funções originais).
    """
    if alg == "nlms":
        errRaw = refBlock - outBlock
        errGrad = errRaw

    elif alg == "dd-lms":
        dec = _decide(outBlock, constSymb)
        errRaw = dec - outBlock
        errGrad = errRaw

    elif alg == "cma":
        errRaw = Rcma - np.abs(outBlock) ** 2
        errGrad = errRaw * outBlock

    elif alg in ("rde", "da-rde"):
        if alg == "rde":
            Rsel = _nearestRadius(outBlock, Rrde)  # raio mais próximo de |y|
        else:
            Rsel = np.abs(refBlock)  # raio exato do símbolo de referência
        errRaw = Rsel**2 - np.abs(outBlock) ** 2
        errGrad = errRaw * outBlock

    elif alg == "static":
        errRaw = np.zeros_like(outBlock)
        errGrad = errRaw

    else:
        raise ValueError(
            "Algoritmo de equalização não especificado (ou incorretamente "
            "especificado) para a versão em domínio da frequência."
        )

    sqErr = np.abs(errRaw) ** 2
    return errGrad, sqErr


def _decide(y, constSymb):
    """Decisor: ponto mais próximo da constelação, por amostra e por modo."""
    dist = np.abs(y[:, :, None] - constSymb[None, None, :]) ** 2
    idx = np.argmin(dist, axis=-1)
    return constSymb[idx]


def _nearestRadius(y, Rrde):
    """Raio da constelação mais próximo de |y|, por amostra e por modo."""
    r = np.abs(y)
    d = np.abs(r[:, :, None] - Rrde[None, None, :])
    idx = np.argmin(d, axis=-1)
    return Rrde[idx]
