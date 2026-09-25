"""
==========================================================================
DSP algorithms for adaptive filtering (:mod:`optic.dsp.adaptiveFiltering`)
==========================================================================

.. autosummary::
   :toctree: generated/
  
   coreAdaptEqBlockTD   -- Time-domain block-wise adaptive equalizer core function.
   coreAdaptEqBlockFD   -- Frequency-domain block-wise adaptive equalizer core function.
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



@njit(fastmath=True, cache=True)
def coreAdaptEqBlockTD(
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
    Adaptive equalizer core processing function (block-wise, time-domain).

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
 
 
def coreAdaptEqBlockFD(
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
    Nfft,
):
    """
    Adaptive equalizer core processing function (block-wise, frequency-domain)
  
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
    Nfft : int
        FFT size used for the overlap-and-save frequency-domain filtering.
        Must satisfy `Nfft >= nTaps`. A power of two is recommended for FFT
        efficiency. Larger `Nfft` amortizes the FFT cost over more valid
        output samples per block, at the cost of a longer coefficient
        "freeze" interval (see Notes).
 
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
 
    Notes
    -----
    **Filtering — overlap-and-save.** For a block starting at symbol index
    `bStart`, the algorithm reads a window of `Nfft` consecutive input
    samples starting at sample `bStart*SpS`, computes its `Nfft`-point
    circular convolution (via FFT) with each zero-padded, time-reversed row
    of the frozen coefficient matrix, and discards the first `nTaps - 1`
    samples of the result (corrupted by circular wrap-around, as usual in
    overlap-and-save). The rows are time-reversed (rather than used as-is)
    because, unlike a conventional causal FIR filter, this equalizer has no
    group delay: output symbol `ind` is formed from input samples `ind*SpS`
    up to `ind*SpS + nTaps - 1` (`coreAdaptEqBlock`'s `indIn = indTaps +
    ind*SpS`), i.e. it looks *forward* rather than backward in time.
 
    The remaining `Nvalid = Nfft - nTaps + 1` samples of the discarded-first
    result are exact, uncorrupted linear-convolution outputs at the full
    (un-decimated) sample rate; every `SpS`-th one of them is a valid
    equalizer output symbol. This yields `Lb = Nvalid // SpS` output symbols
    per block. Any leftover valid samples (when `Nvalid` is not a multiple of
    `SpS`) are simply discarded and transparently recomputed as part of the
    next block's window; consecutive windows overlap whenever `nTaps > SpS`,
    as expected in overlap-and-save.
 
    With `nModes` propagation modes, the MIMO filter is a set of `nModes^2`
    FIR filters, one per (input mode, output mode) pair: `H[m + N*nModes,
    :]` is the response from input mode `N` to output mode `m`. In the
    frequency domain this is nothing more than the classic MIMO channel
    equation evaluated bin by bin, `Y(f) = H(f) X(f)`: for every FFT bin f,
    a small `nModes x nModes` matrix `H(f)` multiplies the `nModes`-vector
    `X(f)`. This is computed for every bin at once with a single
    `np.einsum`, instead of a `nModes x nModes` nested Python loop.
 
    **Adaptation — FFT correlation theorem (LMS family only).** The tap
    update of any LMS-type algorithm (`nlms`, `cma`, `dd-lms`, `rde`,
    `da-rde`) is, at its core, a cross-correlation between an "error-like"
    signal and the equalizer input, evaluated at lags `t = 0..nTaps-1`
    (that is exactly what the time-domain `*UpBlock` functions in
    `blockUpdateFunctions.py` compute via a matrix product). The FFT
    correlation theorem lets that correlation be computed for all `nTaps`
    lags at once from two spectra instead of a `Lb x nTaps` time-domain sum:
    `corr(a, b)[t] = sum_n a[n] conj(b[n+t]) = conj(IFFT(conj(FFT(a)) *
    FFT(b)))[t]`. Since the error only exists at the (sparsely spaced)
    symbol instants, it is first "upsampled" — zero-inserted between
    symbols — before taking its FFT; the input spectrum needed here is the
    very same `Xf` already computed for the filtering step above, so it is
    reused rather than recomputed. This reproduces `nlmsUpBlock`,
    `cmaUpBlock`, `ddlmsUpBlock`, `rdeUpBlock` and `dardeUpBlock`'s gradient
    exactly, without calling those functions (their body *is* the
    time-domain sum) or building `sigInBlock` at all for these algorithms.
 
    `rls` and `dd-rls` have no such shortcut: their state `Sd` (the inverse
    input-correlation matrix) is updated through a recursive
    matrix-inversion-lemma step that is inherently sequential and has no
    simple frequency-domain form, so these two algorithms still build
    `sigInBlock` and call the shared `rlsUpBlock`/`ddrlsUpBlock`, exactly as
    `coreAdaptEqBlock` does. `static` performs no adaptation at all.
    """
    nModes = sigIn.shape[1]
    indTaps = np.arange(nTaps)
 
    errSq = np.zeros((nModes, L))
    sigIn = sigIn.astype(prec)
    H = H.astype(prec)
    H_ = H_.astype(prec)
 
    sigOut = np.zeros((L, nModes), dtype=prec)
 
    if storeCoeff:
        Hiter = np.zeros((nModes**2, nTaps, L), dtype=prec)
    else:
        Hiter = np.zeros((nModes**2, nTaps, 1), dtype=prec)
 
    # RLS inverse-correlation matrix, one nTaps x nTaps identity block per
    # input mode (see rlsUpBlock/ddrlsUpBlock); vectorized with np.tile
    # instead of a concatenation loop.
    Sd = np.tile(np.eye(nTaps, dtype=prec), (nTaps, 1))
 
    # CMA / RDE targets: the Godard radius, and the set of constellation
    # amplitudes ("rings") used as decision radii.
    Rcma = (
        np.mean(np.abs(constSymb) ** 4) / np.mean(np.abs(constSymb) ** 2)
    ) * np.ones((1, nModes)).astype(prec)
    Rrde = np.unique(np.abs(constSymb)).astype(prec)
 
    if Nfft < nTaps:
        raise ValueError("Nfft must be >= nTaps for overlap-and-save filtering.")
 
    Nvalid = Nfft - nTaps + 1  # valid full-rate convolution samples per FFT block
    Lb = max(1, Nvalid // SpS)  # symbols per block, derived from Nfft
 
    nSampAvail = sigIn.shape[0]
    nBlocks = int(np.ceil(L / Lb))
 
    # H reshaped as (input mode, output mode, tap): H3[N, m, :] is the same
    # filter as H[m + N*nModes, :]. Because H is C-contiguous, this reshape
    # is a *view*: writing into H3 updates H in place, with no copy needed.
    H3 = H.reshape(nModes, nModes, nTaps)
    H3_ = H_.reshape(nModes, nModes, nTaps)
 
    for b in range(nBlocks):
        bStart = b * Lb
        bEnd = min(bStart + Lb, L)
        LbCur = bEnd - bStart
 
        # --- coefficients frozen at the start of the block ---
        Hb3 = H3.copy()
        Hb3_ = H3_.copy() if runWL else H3_
 
        globalStart = bStart * SpS
 
        # --- overlap-and-save window (zero-padded near the end of sigIn) ---
        xWin = np.zeros((Nfft, nModes), dtype=prec)
        nAvail = min(Nfft, max(0, nSampAvail - globalStart))
        if nAvail > 0:
            xWin[:nAvail, :] = sigIn[globalStart : globalStart + nAvail, :]
 
        # ==================================================================
        # STEP 1 — filtering: MIMO overlap-and-save via a single batched FFT
        # ==================================================================
        # Spectrum of every input mode, computed in one call (each column of
        # xWin is transformed independently): Xf[N, :] = FFT{x_N}.
        Xf = fft(xWin, Nfft, axis=0).T  # (nModes, Nfft)
 
        # Time-reversed, zero-padded impulse responses of every (N, m) pair,
        # transformed in one call: Hf[N, m, :] = FFT{h_{N->m}[::-1]}.
        HbPadded = np.zeros((nModes, nModes, Nfft), dtype=prec)
        HbPadded[:, :, :nTaps] = Hb3[:, :, ::-1]
        Hf = fft(HbPadded, Nfft, axis=-1)  # (N, m, Nfft)
 
        # MIMO channel equation, bin by bin: Y(f) = H(f) X(f). "Nmf,Nf->mf"
        # contracts over the input-mode axis N, leaving one spectrum per
        # output mode m — the frequency-domain equivalent of the `for N: ...
        # outEq += H[...] @ inEq` accumulation in the time-domain core.
        Yf = np.einsum("Nmf,Nf->mf", Hf, Xf)  # (nModes, Nfft)
 
        if runWL:
            XfConj = fft(xWin.conjugate(), Nfft, axis=0).T
            HbPadded_ = np.zeros((nModes, nModes, Nfft), dtype=prec)
            HbPadded_[:, :, :nTaps] = Hb3_[:, :, ::-1]
            Hf_ = fft(HbPadded_, Nfft, axis=-1)
            Yf = Yf + np.einsum("Nmf,Nf->mf", Hf_, XfConj)
 
        y = ifft(Yf, axis=-1).astype(prec)  # (nModes, Nfft)
        validVals = y[:, nTaps - 1 : nTaps - 1 + Nvalid]  # discard corrupted head
 
        # Decimate every SpS-th (full-rate) sample to recover the symbol-rate
        # equalizer output, for every mode at once.
        outBlock = validVals[:, : LbCur * SpS : SpS].T.copy()  # (LbCur, nModes)
 
        sigOut[bStart:bEnd, :] = outBlock
 
        symbRefBlock = symbRef[bStart:bEnd, :]
 
        # ==================================================================
        # STEP 2 — coefficient update
        # ==================================================================
        if alg in ("nlms", "cma", "dd-lms", "rde", "da-rde"):
            # error-like quantity, decimated rate — matches exactly what
            # each *UpBlock computes before correlating it with the input
            if alg == "nlms":
                err = symbRefBlock - outBlock  # (LbCur, nModes)
                eLike = err
            elif alg == "dd-lms":
                decided = constSymb[_nearestNeighborIndex(outBlock, constSymb)]
                err = decided - outBlock
                eLike = err
            elif alg == "cma":
                err = (Rcma - np.abs(outBlock) ** 2).astype(prec)
                eLike = err * outBlock
            elif alg == "rde":
                decidedR = Rrde[_nearestNeighborIndex(np.abs(outBlock), Rrde)]
                err = (decidedR**2 - np.abs(outBlock) ** 2).astype(prec)
                eLike = err * outBlock
            elif alg == "da-rde":
                decidedR = np.abs(symbRefBlock).astype(prec)
                err = (decidedR**2 - np.abs(outBlock) ** 2).astype(prec)
                eLike = err * outBlock
 
            # "Upsample" eLike back to the full sample rate (zeros between
            # symbol instants) so it can be correlated, via FFT, with the
            # full-rate input spectrum Xf.
            E = np.zeros((nModes, Nfft), dtype=prec)
            E[:, : LbCur * SpS : SpS] = eLike.T
            Ef = fft(E, Nfft, axis=-1)  # (m, Nfft)
 
            if alg == "nlms":
                # NLMS normalizes each symbol's contribution by the input
                # energy in its own tap window; that window depends on the
                # input mode N, so (unlike the other algorithms) the
                # normalized error can't be shared across N. A sliding-window
                # sum of |x|^2 via cumsum avoids rebuilding sigInBlock.
                power = np.abs(xWin) ** 2  # (Nfft, nModes)
                cumPower = np.vstack([np.zeros(nModes), np.cumsum(power, axis=0)])
                k = np.arange(LbCur)
                normPerSample = cumPower[k * SpS + nTaps, :] - cumPower[k * SpS, :]
                # (LbCur, nModes) -> one norm curve per input mode N
 
                for N in range(nModes):
                    En = np.zeros((nModes, Nfft), dtype=prec)
                    En[:, : LbCur * SpS : SpS] = (eLike / normPerSample[:, N : N + 1]).T
                    Efn = fft(En, Nfft, axis=-1)
 
                    grad = np.conj(ifft(np.conj(Efn) * Xf[N], axis=-1))[:, :nTaps]
                    H3[N, :, :] += mu * grad
                    if runWL:
                        Efn_conj = fft(np.conj(En), Nfft, axis=-1)
                        gradWL = ifft(np.conj(Efn_conj) * Xf[N], axis=-1)[:, :nTaps]
                        H3_[N, :, :] += mu * gradWL
            else:
                # eLike (and therefore its spectrum Ef) does not depend on
                # the input mode N here, so the correlation with every Xf[N]
                # is a single broadcasted FFT/IFFT pair for the whole
                # (N, m) grid at once — no loop at all.
                term = np.conj(Ef)[None, :, :] * Xf[:, None, :]  # (N, m, Nfft)
                grad = np.conj(ifft(term, axis=-1))[:, :, :nTaps]
                H3 += mu * grad
                if runWL:
                    EfConj = fft(np.conj(E), Nfft, axis=-1)
                    termWL = np.conj(EfConj)[None, :, :] * Xf[:, None, :]
                    gradWL = ifft(termWL, axis=-1)[:, :, :nTaps]
                    H3_ += mu * gradWL
 
            errBlock = (np.abs(err) ** 2).T
 
        elif alg in ("rls", "dd-rls"):
            # no simple frequency-domain equivalent (see Notes): Sd is a
            # time-domain object, so this still calls the shared functions
            indInBlock = (bStart + np.arange(LbCur))[:, None] * SpS + indTaps[None, :]
            sigInBlock = sigIn[indInBlock, :]  # (LbCur, nTaps, nModes)
 
            if alg == "rls":
                H, Sd, errBlock = rlsUpBlock(
                    sigInBlock, symbRefBlock, outBlock, lambdaRLS, H, Sd, nModes, prec
                )
            else:
                H, Sd, errBlock = ddrlsUpBlock(
                    sigInBlock, constSymb, outBlock, lambdaRLS, H, Sd, nModes, prec
                )
            # rlsUpBlock/ddrlsUpBlock mutate H in place and return that same
            # object, so H3 (a view created before the block loop) already
            # reflects the update; no explicit refresh is needed here.
 
        elif alg == "static":
            prevErr = errSq[:, bStart - 1] if bStart > 0 else np.zeros(nModes)
            errBlock = np.tile(prevErr[:, None], (1, LbCur))
 
        else:
            raise ValueError(
                "Equalization algorithm not specified (or incorrectly specified)."
            )
 
        errSq[:, bStart:bEnd] = errBlock
 
        if storeCoeff:
            Hiter[:, :, bStart:bEnd] = H[:, :, None]
        else:
            Hiter[:, :, 0] = H
 
    return sigOut, H, H_, errSq, Hiter


def _nearestNeighborIndex(values, alphabet):
    """
    Slicer function for nearest-neighbor quantization of `values` to the
    constellation `alphabet`. Returns an array of the same shape as `values`
    containing the indices of the nearest constellation points in `alphabet`

    Parameters
    ----------
    values : np.array
        Input values to be quantized.
    alphabet : np.array
        Constellation points to which `values` will be quantized.
        Must be 1-D.
    
    Returns
    -------
    np.array
        Indices of the nearest constellation points in `alphabet` for each
        element in `values`.
    """
    dist = np.abs(values[..., None] - alphabet[(None,) * values.ndim + (slice(None),)])
    return np.argmin(dist, axis=-1)