"""
================================================================
Forward error correction (FEC) utilities (:mod:`optic.comm.fec`)
================================================================

.. autosummary::
   :toctree: generated/

   par2gen                -- Parity-check matrix to generator matrix conversion
   gaussElim              -- Gaussian elimination over GF(2)
   encoder                -- Performs linear block encoding
   encodeDVBS2            -- Encode binary sequences using a DVB-S2 LDPC parity-check matrix
   encodeTriang           -- Encode binary sequences using lower-triangular parity-check matrices
   sumProductAlgorithm    -- Belief propagation decoding using the sum-product algorithm
   minSumAlgorithm        -- Belief propagation decoding using the min-sum algorithm
   encodeLDPC             -- Encode binary sequences using a LDPC parity-check matrix
   decodeLDPC             -- Decode multiple LDPC codewords using belief propagation
   writeAlist             -- Save a binary parity-check matrix to ALIST format
   readAlist              -- Read an ALIST file and reconstruct the binary parity-check matrix
   triangularize          -- Convert binary matrix to lower-triangular form
   triangP1P2             -- Extract matrices that compute parities from lower-triangular form H
   inverseMatrixGF2       -- Invert a square binary matrix over GF(2)
   plotBinaryMatrix       -- Plot a binary matrix using matplotlib
   parseAlist             -- Parse an ALIST file and extract the code parameters
   summarizeAlistFolder   -- Summarize ALIST files in a folder in a table
"""

"""Forward error correction (FEC) utilities."""
import logging as logg

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
import os
from prettytable import PrettyTable


def par2gen(H):
    """
    Convert a binary parity-check matrix H into a systematic generator matrix G over GF(2).

    Parameters
    ----------
    H : ndarray of shape (n - k, n)
        Parity-check matrix with entries in {0, 1}.

    Returns
    -------
    G : ndarray of shape (k, n)
        Systematic generator matrix corresponding to the input parity-check matrix H.
        The form of G is :math:`[I_k | P]`, where :math:`I_k` is the identity matrix and :math:`P` is a binary matrix.
    colSwaps : ndarray of shape (n,)
        Indices representing the column permutations applied to H to obtain Hm.
        These permutations are required to match the systematic form used in G.
    Hm : ndarray of shape (n - k, n)
        The modified parity-check matrix after Gaussian elimination and column reordering.
        This matrix has the identity portion on the right-hand side, corresponding to
        a standard systematic form :math:`[P^T | I_{n-k}]`.

    Notes
    -----
    This method assumes that the parity-check matrix H is full rank and that the identity
    portion can be isolated on the right via column permutations.
    """
    n = H.shape[1]
    k = n - H.shape[0]

    if type(H) == csr_matrix:
        H = csr_matrix.todense(H).astype(np.uint8)
    elif type(H) == csc_matrix:
        H = csc_matrix.todense(H).astype(np.uint8)
    elif type(H) == coo_matrix:
        H = coo_matrix.todense(H).astype(np.uint8)

    Hs = gaussElim(H)  # Reduce matrix to row echelon form
    Hs = np.array(Hs, dtype=np.uint8)

    # do the necessary column swaps
    cols = np.arange(Hs.shape[1])
    indP = cols[
        (np.sum(Hs[:, 0:], axis=0) > 1)
    ]  # indexes of cols belonging to matrix P
    indI = cols[
        (np.sum(Hs[:, 0:], axis=0) == 1)
    ]  # indexes of cols belonging to the indentity

    Hm = np.hstack((Hs[:, cols[indP]], Hs[:, cols[indI]]))

    colSwaps = np.hstack((indP, indI))

    # systematic generator matrix G
    G = np.hstack((np.eye(k, dtype=np.uint8), Hm[:, 0:k].T))

    return G, colSwaps, H[:, colSwaps]


@njit
def gaussElim(M):
    """
    Perform Gaussian elimination over GF(2) to reduce a binary matrix to row echelon form.

    Parameters
    ----------
    M : ndarray of uint8
        Input binary matrix (dense, 2D). All operations are over GF(2).

    Returns
    -------
    matrix : ndarray of uint8
        Matrix in row echelon form (mod 2).
    """
    rowCount, columnCount = M.shape
    lead = 0

    matrix = M.copy()
    for r in range(rowCount):
        if lead >= columnCount:
            break

        i = r
        while matrix[i, lead] == 0:
            i += 1
            if i == rowCount:
                i = r
                lead += 1
                if lead == columnCount:
                    break
        if lead == columnCount:
            break

        # Swap rows r and i manually
        for k in range(columnCount):
            temp = matrix[r, k]
            matrix[r, k] = matrix[i, k]
            matrix[i, k] = temp

        # Eliminate other rows
        for j in range(rowCount):
            if j != r and matrix[j, lead] == 1:
                for k in range(columnCount):
                    matrix[j, k] ^= matrix[r, k]  # XOR for GF(2)

        lead += 1

    return matrix


def encodeLDPC(bits, param):
    """
    Encode binary sequences using a parity-check matrix of a LDPC code.

    Parameters
    ----------
    bits : ndarray of shape (k, N)
        Binary input sequences to be encoded. Each column is a bit sequence of length :math:`k` bits.
    param : object
        Object containing the following attributes:

        - mode : str
            Mode of operation ('DVBS2', 'IEEE_802.11nD2', or 'AR4JA').

        - H : ndarray of shape (n - k, n)
            Binary parity-check matrix :math:`H`.

        - G : ndarray of shape (k, n), optional
            Binary generator matrix :math:`G`.

        - systematic : bool, optional
            If True, the generator matrix is assumed to be in systematic form. If False,
            the generator matrix is treated as a general linear transformation (default is True).

        - P1 : ndarray of shape (m, k), optional
            Matrix used for encoding in triangular mode.

        - P2 : ndarray of shape (m, k), optional
            Matrix used for encoding in triangular mode.

    Returns
    -------
    codewords : ndarray of shape (n, N)
        Binary encoded codewords. Each column is a codeword of length :math:`n` corresponding
        to the respective input bit sequence.

    References
    ----------
    [1] T. J. Richardson and R. L. Urbanke, "Efficient encoding of low-density parity-check codes," IEEE Transactions on Information Theory, vol. 47, no. 2, pp. 638-656, Feb 2001.

    """
    # check input parameters
    mode = getattr(param, "mode", "DVBS2")
    n = getattr(param, "n", 64800)
    R = getattr(param, "R", "4/5")
    H = getattr(param, "H", None)
    G = getattr(param, "G", None)
    systematic = getattr(param, "systematic", True)
    P1 = getattr(param, "P1", None)
    P2 = getattr(param, "P2", None)
    path = getattr(param, "path", None)

    if H is None:
        try:
            filename = f"LDPC_{mode}_{n}b_R{R[0]}{R[2]}.txt"
            H = readAlist(path + filename)
            param.H = H
        except FileNotFoundError:
            logg.error(
                f"File {filename} not found. Please provide a valid parity-check matrix H."
            )

    if mode == "DVBS2":
        k = n - H.shape[0]
        A = csr_matrix.todense(H[:, :k]).astype(np.uint8)
        if H is None:
            raise ValueError(
                "H is None. Please provide a valid DVBS2 parity-check matrix."
            )
        return encodeDVBS2(bits, A)
    elif mode == "IEEE_802.11nD2":
        if P1 is None or P2 is None:
            # attempt to triangularize H
            P1, P2, Hm = triangP1P2(H) 
            if P1 is None or P2 is None:
                # if H cannot be triangularized, encode with G
                if G is None:
                    G, _, Hm = par2gen(H)  # get systematic generator matrix G
                    G = G.astype(np.uint8)
                    param.G = G
                    param.H = csr_matrix(Hm)
                    codedBits = encoder(G, bits, systematic)
                    return codedBits[0 : param.n, :]
                else:
                    G = G.astype(np.uint8)
                    codedBits = encoder(G, bits, systematic)
                    return codedBits[0 : param.n, :]                     
            else:           
                # encode with triangularized H      
                param.P1 = P1
                param.P2 = P2
                param.H = csr_matrix(Hm)
                return encodeTriang(bits, P1, P2)
    elif mode == "AR4JA":
        if G is None:
            G, _, Hm = par2gen(H)  # get systematic generator matrix G
            G = G.astype(np.uint8)
            param.G = G
            param.H = csr_matrix(Hm)
            codedBits = encoder(G, bits, systematic)
            return codedBits[0 : param.n, :]
        else:
            G = G.astype(np.uint8)
            codedBits = encoder(G, bits, systematic)
            return codedBits[0 : param.n, :]
    else:
        logg.error(
            f"Unsupported mode: {mode}. Supported modes are: DVBS2, IEEE_802.11nD2, AR4JA."
        )


@njit(parallel=True)
def encodeDVBS2(bits, A):
    """
    Encode multiple binary sequences using a DVB-S2 LDPC parity-check matrix.

    Parameters
    ----------
    bits : ndarray of shape (k, N)
        Binary input sequences to be encoded. Each column represents a bit sequence of length :math:`k`.
    A : ndarray of shape (m, k)
        Matrix corresponding to the first :math:`k` columns of the parity-check matrix :math:`H`.

    Returns
    -------
    codewords : ndarray of shape (n, N)
        Binary encoded codewords. Each column is a codeword of length :math:`n` corresponding
        to the respective input bit sequence.
    """
    A = A.astype(np.uint8)
    bits = bits.astype(np.uint8)

    m, k = A.shape
    n = k + m
    N = bits.shape[1]

    codewords = np.zeros((n, N), dtype=np.uint8)

    for col in prange(N):
        # Copy bit sequence bits
        for i in range(k):
            codewords[i, col] = bits[i, col]

        # Compute parity bits
        parity = np.zeros(m, dtype=np.uint8)
        for i in range(m):
            acc = 0
            for j in range(k):
                acc ^= A[i, j] & bits[j, col]
            parity[i] = acc

        # Recursive parity encoding
        codewords[k, col] = parity[0]
        for i in range(1, m):
            codewords[k + i, col] = parity[i] ^ codewords[k + i - 1, col]

    return codewords


@njit(parallel=True)
def encoder(G, bits, systematic=True):
    """
    Encode binary sequences using a generator matrix over GF(2).

    Parameters
    ----------
    G : ndarray of shape (k, n)
        Binary generator matrix.
    bits : ndarray of shape (k, N)
        Binary input sequences to encode. Each column is a bit sequence of length :math:`k`.
    systematic : bool, optional
        If True, the generator matrix is assumed to be in systematic form. If False, the
        generator matrix is treated as a general linear transformation (default is True).

    Returns
    -------
    codewords : ndarray of shape (n, N)
        Encoded binary codewords. Each column corresponds to a codeword of length :math:`n` resulting
        from applying the generator matrix to the respective input bit sequence.
    """
    G = G.T
    n, k = G.shape
    _, N = bits.shape
    codewords = np.zeros((n, N), dtype=np.uint8)

    if systematic:
        codewords[:k, :] = bits
        for col in prange(N):  # for each input word
            for i in range(k, n):  # for each codeword bit
                acc = 0
                for j in range(k):
                    acc ^= G[i, j] & bits[j, col]  # binary dot product
                codewords[i, col] = acc
    else:
        for col in prange(N):
            for i in range(n):
                acc = 0
                for j in range(k):
                    acc ^= G[i, j] & bits[j, col]
                codewords[i, col] = acc

    return codewords


@njit(parallel=True, fastmath=True)
def sumProductAlgorithm(llrs, checkNodes, varNodes, maxIter, prec=np.float32):
    """
    Performs belief propagation decoding using the sum-product algorithm (SPA) for multiple codewords.

    Parameters
    ----------
    llrs : ndarray of shape (n, numCodewords)
        Array of log-likelihood ratios (LLRs) for each bit of the received codeword.
    checkNodes : list of ndarray
        List of length :math:`m`, where each element is a 1D array containing the indices
        of variable nodes (bits) involved in the corresponding check node (parity-check equation).
    varNodes : list of ndarray
        List of length :math:`n`, where each element is a 1D array containing the indices
        of check nodes that the corresponding variable node participates in.
    maxIter : int
        Maximum number of belief propagation iterations.
    prec : data-type, optional
        Data type for the computations (default is np.float32).

    Returns
    -------
    finalLLR : ndarray of shape (n,)
        Updated log-likelihood ratios after message passing.
    numIter : int
        Number of iterations executed until decoding converged or reached `maxIter`.
    frameDecodingFail : ndarray of shape (numCodewords,)
        Array indicating whether decoding was successful (0) or failed (1) for each codeword.
        A value of 0 indicates successful decoding, while 1 indicates failure.

    References
    ----------
    [1] F. R. Kschischang, B. J. Frey and H. . -A. Loeliger, "Factor graphs and the sum-product algorithm," IEEE Transactions on Information Theory, vol. 47, no. 2, pp. 498-519, Feb 2001.

    [2] T. J. Richardson and R. L. Urbanke, "The capacity of low-density parity-check codes under message-passing decoding," IEEE Transactions on Information Theory, vol. 47, no. 2, pp. 599-618, Feb 2001.
    """
    m, n = len(checkNodes), len(varNodes)
    msgVtoC = np.zeros((m, n), dtype=prec)
    msgCtoV = np.zeros((m, n), dtype=prec)
    llrs = llrs.astype(prec)

    numCodewords = llrs.shape[1]
    finalLLR = np.zeros((n, numCodewords), dtype=prec)
    frameDecodingFail = np.ones((numCodewords,), dtype=np.int8)
    lastIter = np.zeros((numCodewords,), dtype=np.uint32)

    for indCw in range(numCodewords):
        decodedBits = np.zeros(n, dtype=np.uint8)
        llr = llrs[:, indCw]
        # Initialize variable-to-check messages with input LLRs
        for var in prange(n):
            for check in varNodes[var]:
                msgVtoC[check, var] = llr[var]

        for indIter in range(maxIter):
            # Check-to-variable update
            for check in prange(m):
                for var_idx in range(len(checkNodes[check])):
                    var = checkNodes[check][var_idx]
                    product = 1.0
                    for neighbor_idx in range(len(checkNodes[check])):
                        neighbor = checkNodes[check][neighbor_idx]
                        if neighbor != var:
                            product *= np.tanh(msgVtoC[check, neighbor] / 2)
                    product = min(0.999999, max(-0.999999, product))  # clip
                    msgCtoV[check, var] = 2 * np.arctanh(product)

            # Variable-to-check update
            for var in prange(n):
                for check_idx in range(len(varNodes[var])):
                    check = varNodes[var][check_idx]
                    sumMsg = llr[var]
                    for neighbor_idx in range(len(varNodes[var])):
                        neighbor = varNodes[var][neighbor_idx]
                        if neighbor != check:
                            sumMsg += msgCtoV[neighbor, var]
                    msgVtoC[check, var] = sumMsg

            # Final LLR computation
            for var in prange(n):
                finalLLR[var, indCw] = llr[var]
                for check in varNodes[var]:
                    finalLLR[var, indCw] += msgCtoV[check, var]
                    decodedBits[var] = (-np.sign(finalLLR[var, indCw]) + 1) // 2

            # Check parity conditions
            parity_checks = np.zeros(m, dtype=np.uint8)
            for indParity in prange(m):
                for check in checkNodes[indParity]:
                    parity_checks[indParity] ^= decodedBits[check]  # accumulate XORs

            if np.sum(parity_checks) == 0:
                frameDecodingFail[indCw] = 0
                lastIter[indCw] = indIter
                break

            if indIter == maxIter - 1:
                lastIter[indCw] = indIter

    return finalLLR, lastIter, frameDecodingFail


@njit(parallel=True, fastmath=True)
def minSumAlgorithm(llrs, checkNodes, varNodes, maxIter, prec=np.float32):
    """
    Performs belief propagation decoding using the Min-Sum Algorithm (MSA) for multiple codewords.

    Parameters
    ----------
    llrs : ndarray of shape (n, numCodewords)
        Log-likelihood ratios (LLRs) of the received codeword bits.
    checkNodes : list of ndarray
        List of length :math:`m`, where each entry contains the indices of variable nodes
        connected to the corresponding check node.
    varNodes : list of ndarray
        List of length :math:`n`, where each entry contains the indices of check nodes
        connected to the corresponding variable node.
    maxIter : int
        Maximum number of iterations for belief propagation.
    prec : data-type, optional
        Numerical precision to use in computations (default is np.float32).

    Returns
    -------
    finalLLR : ndarray of shape (n,)
        Updated LLR values for the decoded codeword after the final iteration.
    numIter : int
        Number of iterations performed before successful decoding or reaching `maxIter`.
    frameDecodingFail : ndarray of shape (numCodewords,)
        Array indicating whether decoding was successful (0) or failed (1) for each codeword.
        A value of 0 indicates successful decoding, while 1 indicates failure.

    References
    ----------
    [1] M. P. C. Fossorier, M. Mihaljevic and H. Imai, "Reduced complexity iterative decoding of low-density parity check codes based on belief propagation," IEEE Transactions on Communications, vol. 47, no. 5, pp. 673-680, May 1999
    """
    m, n = len(checkNodes), len(varNodes)
    msgVtoC = np.zeros((m, n), dtype=prec)
    msgCtoV = np.zeros((m, n), dtype=prec)
    llrs = llrs.astype(prec)

    numCodewords = llrs.shape[1]
    finalLLR = np.zeros((n, numCodewords), dtype=prec)
    frameDecodingFail = np.ones((numCodewords,), dtype=np.int8)
    lastIter = np.zeros((numCodewords,), dtype=np.uint32)

    for indCw in range(numCodewords):
        decodedBits = np.zeros(n, dtype=np.uint8)
        llr = llrs[:, indCw]

        # Initialize variable-to-check messages with input LLRs
        for var in prange(n):
            for check in varNodes[var]:
                msgVtoC[check, var] = llr[var]

        for indIter in range(maxIter):
            # Check-to-variable update (Min-Sum)
            for check in prange(m):
                for var in checkNodes[check]:
                    signProduct = 1
                    min_abs = np.inf
                    for neighbor in checkNodes[check]:
                        if neighbor != var:
                            val = msgVtoC[check, neighbor]
                            signProduct *= np.sign(val)
                            min_abs = min(min_abs, abs(val))
                    msgCtoV[check, var] = signProduct * min_abs

            # Variable-to-check update
            for var in prange(n):
                for check in varNodes[var]:
                    sumMsg = llr[var]
                    for neighbor in varNodes[var]:
                        if neighbor != check:
                            sumMsg += msgCtoV[neighbor, var]
                    msgVtoC[check, var] = sumMsg

            # Final LLR and decision
            for var in prange(n):
                finalLLR[var, indCw] = llr[var]
                for check in varNodes[var]:
                    finalLLR[var, indCw] += msgCtoV[check, var]
                decodedBits[var] = (-np.sign(finalLLR[var, indCw]) + 1) // 2

            # Check parity conditions
            parity_checks = np.zeros(m, dtype=np.uint8)
            for indParity in prange(m):
                for check in checkNodes[indParity]:
                    parity_checks[indParity] ^= decodedBits[check]  # accumulate XORs

            if np.sum(parity_checks) == 0:
                frameDecodingFail[indCw] = 0
                lastIter[indCw] = indIter
                break

            if indIter == maxIter - 1:
                lastIter[indCw] = indIter

    return finalLLR, lastIter, frameDecodingFail


def decodeLDPC(llrs, param):
    """
    Decode multiple LDPC codewords using the belief propagation algorithms.

    Parameters
    ----------
    llrs : ndarray of shape (n, numCodewords)
        Array of log-likelihood ratios (LLRs) for each bit of the received codewords.
        Codewords are assumed to be disposed in columns.
    param : object
        Object containing the following attributes:

        - H : ndarray of shape (m, n)
            Sparse binary parity-check matrix of the LDPC code.

        - maxIter : int
            Maximum number of iterations for belief propagation.

        - alg : str
            Decoding algorithm to use ('SPA' for sum-product or 'MSA' for min-sum).

        - prec : data-type
            Numerical precision to use in computations (default is np.float32).

    Returns
    -------
    decodedBits : ndarray of shape (n, numCodewords)
        Array of decoded bits for each codeword.
    outputLLRs : ndarray of shape (n, numCodewords)
        Array of updated log-likelihood ratios (LLRs) after decoding.
    """
    # check input parameters
    H = getattr(param, "H", None)
    maxIter = getattr(param, "maxIter", 25)
    alg = getattr(param, "alg", "SPA")
    prec = getattr(param, "prec", np.float32)

    if H is None:
        logg.error("H is None. Please provide a valid parity-check matrix.")

    m, n = H.shape
    numCodewords = llrs.shape[1]
    n_ = llrs.shape[0]
    Hcsc = H.tocsc()  # convert to CSC format for efficient column access

    llrs = np.clip(llrs, -200, 200)
    outputLLRs = np.zeros_like(llrs, dtype=prec)

    # depuncturing LLRs if necessary
    if n_ < n:
        llrs = np.pad(llrs, ((0, n - n_), (0, 0)), mode="constant")

    # Build adjacency lists using fixed-size lists for Numba
    checkNodes = List([H[i].indices.astype(np.uint32) for i in range(m)])
    varNodes = List([Hcsc[:, j].indices.astype(np.uint32) for j in range(n)])

    logg.info(f"Decoding {numCodewords} LDPC codewords with {alg}")
    if alg == "SPA":
        outputLLRs, lastIter, frameErrors = sumProductAlgorithm(
            llrs, checkNodes, varNodes, maxIter, prec
        )
    elif alg == "MSA":
        outputLLRs, lastIter, frameErrors = minSumAlgorithm(
            llrs, checkNodes, varNodes, maxIter, prec
        )
    else:
        logg.error(f"Unsupported algorithm: {alg}. Supported algorithms are: SPA, MSA.")
        return None, None

    for indCw, frameError in enumerate(frameErrors):
        if frameError == 0:
            logg.info(
                f"Frame {indCw} - Successful decoding at iteration {lastIter[indCw]}."
            )

    # remove punctured bits if necessary
    if n_ < n:
        outputLLRs = outputLLRs[0:n_, :]

    decodedBits = ((-np.sign(outputLLRs) + 1) // 2).astype(np.int8)

    return decodedBits, outputLLRs, frameErrors


def writeAlist(H, filename):
    """
    Save a binary parity-check matrix H (numpy array) to ALIST format.

    Parameters
    ----------
    H : ndarray of shape (m, n)
        Binary parity-check matrix.
    filename : str
        Name of the ALIST file to be written.
    """
    if type(H) == csr_matrix:
        H = csr_matrix.todense(H).astype(np.uint8)
    elif type(H) == csc_matrix:
        H = csc_matrix.todense(H).astype(np.uint8)
    elif type(H) == coo_matrix:
        H = coo_matrix.todense(H).astype(np.uint8)
    else:
        H = H.astype(np.int8)

    m, n = H.shape

    # Variable and check node degrees
    varDegrees = [int(np.sum(H[:, j])) for j in range(n)]
    checkDegrees = [int(np.sum(H[i, :])) for i in range(m)]
    maxColDeg = max(varDegrees)
    maxRowDeg = max(checkDegrees)

    with open(filename, "w") as f:
        f.write(f"{n} {m}\n")
        f.write(f"{maxColDeg} {maxRowDeg}\n")

        f.write(" ".join(str(d) for d in varDegrees) + "\n")
        f.write(" ".join(str(d) for d in checkDegrees) + "\n")

        # Variable node connections (1-based indexing)
        for j in range(n):
            connections = np.where(H[:, j] == 1)[0] + 1
            padded = list(connections) + [0] * (maxColDeg - len(connections))
            f.write(" ".join(str(i) for i in padded) + "\n")

        # Check node connections (1-based indexing)
        for i in range(m):
            connections = np.where(H[i, :] == 1)[1] + 1
            padded = list(connections) + [0] * (maxRowDeg - len(connections))
            f.write(" ".join(str(j) for j in padded) + "\n")

    f.close()


def readAlist(filename):
    """
    Read an ALIST file and reconstruct the binary parity-check matrix H.

    Parameters
    ----------
    filename : str
        Path to the ALIST file.

    Returns
    -------
    H : ndarray of shape (m, n)
        Reconstructed binary parity-check matrix.
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n, m = map(int, lines[0].split())
    var_conn_lines = lines[4 : 4 + n]

    H = np.zeros((m, n), dtype=np.uint8)

    for j, line in enumerate(var_conn_lines):
        for entry in map(int, line.split()):
            if entry > 0:
                H[entry - 1, j] = 1

    return csr_matrix(H)


@njit
def inverseMatrixGF2(A):
    """
    Invert a square binary matrix over GF(2) using Gauss-Jordan elimination.

    Parameters
    ----------
    A : ndarray of shape (n, n), dtype=np.uint8
        Binary square matrix.

    Returns
    -------
    Ainv : ndarray of shape (n, n), dtype=np.uint8
        Inverse of A over GF(2), if invertible. If not invertible, returns the identity matrix.
    success : bool
        True if A is invertible, False otherwise.
    """
    n = A.shape[0]
    A = A.copy()
    Ainv = np.zeros((n, n), dtype=np.uint8)  # Initialize inverse matrix
    for i in range(n):
        Ainv[i, i] = 1  # identity matrix

    for i in range(n):
        # Find pivot
        if A[i, i] == 0:
            found = False
            for j in range(i + 1, n):
                if A[j, i] == 1:
                    # Manually swap rows i and j in A and Ainv
                    for k in range(n):
                        tmp = A[i, k]
                        A[i, k] = A[j, k]
                        A[j, k] = tmp
                        tmp = Ainv[i, k]
                        Ainv[i, k] = Ainv[j, k]
                        Ainv[j, k] = tmp
                    found = True
                    break
            if not found:
                return Ainv, False  # Matrix is not invertible

        # Eliminate all other entries in column i
        for j in range(n):
            if j != i and A[j, i] == 1:
                for k in range(n):
                    A[j, k] ^= A[i, k]
                    Ainv[j, k] ^= Ainv[i, k]

    return Ainv, True


@njit
def triangularize(H):
    """
    Convert binary matrix H into lower-triangular form using only row and column permutations.

    Parameters
    ----------
    H : ndarray of shape (m, n), dtype=np.uint8
        Binary parity-check matrix :math:`H`.

    Returns
    -------
    triangH : ndarray of shape (m, n), dtype=np.uint8
        Triangularized matrix.
    rowPerm : ndarray of shape (m,), dtype=np.int32
        Row permutation indices.
    colPerm : ndarray of shape (n,), dtype=np.int32
        Column permutation indices.

    References
    ----------
    [1] T. J. Richardson and R. L. Urbanke, "Efficient encoding of low-density parity-check codes," IEEE Transactions on Information Theory, vol. 47, no. 2, pp. 638-656, Feb 2001.
    """
    m, n = H.shape
    triangH = H.copy()
    rowPerm = np.arange(m, dtype=np.int32)
    colPerm = np.arange(n, dtype=np.int32)

    for i in range(m):
        pivotFound = False

        for r in range(i, m):
            for c in range(i, n):
                if triangH[r, c] == 1:
                    # Swap rows
                    if r != i:
                        for j in range(n):
                            triangH[i, j], triangH[r, j] = triangH[r, j], triangH[i, j]
                        rowPerm[i], rowPerm[r] = rowPerm[r], rowPerm[i]
                    # Swap columns
                    if c != i:
                        for j in range(m):
                            triangH[j, i], triangH[j, c] = triangH[j, c], triangH[j, i]
                        colPerm[i], colPerm[c] = colPerm[c], colPerm[i]
                    pivotFound = True
                    break
            if pivotFound:
                break

        if not pivotFound:
            # Leave zeros if no pivot found in this column
            continue

        # Eliminate below
        for r in range(i + 1, m):
            if triangH[r, i] == 1:
                for j in range(n):
                    triangH[r, j] ^= triangH[i, j]

    return triangH, rowPerm, colPerm


def triangP1P2(H):
    """
    Convert a binary parity-check matrix H into a lower-triangular form and extract matrices P1 and P2.

    Parameters
    ----------
    H : ndarray of shape (m, n)
        Binary parity-check matrix. It is used to derive the matrices P1 and P2.

    Returns
    -------
    P1 : ndarray of shape (m1, k)
        First parity matrix.
    P2 : ndarray of shape (m2, k)
        Second parity matrix.
    triangH : ndarray of shape (m, n)
        Triangularized H matrix.

    References
    ----------
    [1] T. J. Richardson and R. L. Urbanke, "Efficient encoding of low-density parity-check codes," IEEE Transactions on Information Theory, vol. 47, no. 2, pp. 638-656, Feb 2001.
    """
    H = csr_matrix.todense(H).astype(np.uint8)

    # convert to lower-triangular form
    triangH, _, colSwaps = triangularize(H)

    # calculate the gap g
    idx = np.where(triangH[:, -1] == 1)
    g = triangH.shape[0] - np.min(idx[0]) - 1
    m = triangH.shape[0]
    n = triangH.shape[1]
    k = n - m

    # extract matrices
    E = triangH[m - g :, n - (m - g) :]
    T = triangH[0 : m - g, n - (m - g) :]
    A = triangH[0 : m - g, 0:k]
    B = triangH[0 : m - g, k : k + g]
    C = triangH[m - g :, 0:k]
    D = triangH[m - g :, k : k + g]

    # invert matrix T
    T_inv, found = inverseMatrixGF2(T)
    if not found:
        logg.warning("Matrix T is not invertible.")
        return None, None, None

    X = np.mod(E @ T_inv, 2)
    C_tilde = np.mod(X @ A + C, 2)
    D_tilde = np.mod(X @ B + D, 2)

    # invert matrix D tilde
    D_tilde_inv, found = inverseMatrixGF2(D_tilde)
    if not found:
        logg.warning("Matrix D_tilde is not invertible.")
        return None, None, None

    P1 = np.mod(D_tilde_inv @ C_tilde, 2)
    P2 = np.mod(T_inv @ np.mod(A + np.mod(B @ P1, 2), 2), 2)

    return P1, P2, H[:, colSwaps]


@njit(parallel=True)
def encodeTriang(bits, P1, P2):
    """
    Encode binary sequences using two parity matrices for LDPC encoding.

    Parameters
    ----------
    bits : ndarray of shape (k, N)
        Binary input sequences. Each column is a bit sequence to be encoded.
    P1 : ndarray of shape (m1, k)
        First parity matrix.
    P2 : ndarray of shape (m2, k)
        Second parity matrix.

    Returns
    -------
    codewords : ndarray of shape (k + m1 + m2, N)
        Encoded codewords, one per column.

    References
    ----------
    [1] T. J. Richardson and R. L. Urbanke, "Efficient encoding of low-density parity-check codes," IEEE Transactions on Information Theory, vol. 47, no. 2, pp. 638-656, Feb 2001.
    """
    bits = bits.astype(np.uint8)
    m1 = P1.shape[0]
    m2 = P2.shape[0]
    k = bits.shape[0]
    N = bits.shape[1]
    n = k + m1 + m2

    codewords = np.zeros((n, N), dtype=np.uint8)

    # Copy message bits
    for col in prange(N):
        for i in range(k):
            codewords[i, col] = bits[i, col]

    # Compute first parity section
    for col in prange(N):
        for i in range(m1):
            acc = 0
            for j in range(k):
                acc ^= P1[i, j] & bits[j, col]
            codewords[k + i, col] = acc

    # Compute second parity section
    for col in prange(N):
        for i in range(m2):
            acc = 0
            for j in range(k):
                acc ^= P2[i, j] & bits[j, col]
            codewords[k + m1 + i, col] = acc

    return codewords


def plotBinaryMatrix(H):
    """
    Plot the binary matrix H with dots at positions where H[i,j] = 1.

    Parameters
    ----------
    H : ndarray of shape (m, n)
        Binary matrix.
    """
    H = np.asarray(H)
    rows, cols = np.where(H == 1)
    plt.scatter(cols, rows, s=0.05, color="blue")  # s controls dot size
    plt.gca().invert_yaxis()
    plt.xlabel("Column indexes")
    plt.ylabel("Row indexes")
    plt.title(f"Matrix: {H.shape[0]} $\\times$ {H.shape[1]}")
    plt.axis("square")
    plt.xlim(0, H.shape[1])
    plt.ylim(H.shape[0], 0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def parseAlist(path):
    """
    Parse an LDPC ALIST file and return basic parameters.

    Parameters
    ----------
    path : str
        Path to the folder with the ALIST files.

    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Read header: n, m
    n, m = map(int, lines[0].split())

    # Skip variable/check node degree lists (lines 2 to 1+n+m)
    offset = 2 + n + m

    # Variable node connections: shape (n, max_col_w)
    var_nodes = [list(map(int, lines[i].split())) for i in range(2, 2 + n)]

    # Check node connections: shape (m, max_row_w)
    check_nodes = [list(map(int, lines[i].split())) for i in range(2 + n, offset)]

    # Optional sanity check: number of connections per node
    col_weights = np.array([len([v for v in row if v > 0]) for row in var_nodes])
    row_weights = np.array([len([v for v in row if v > 0]) for row in check_nodes])

    rate = (n - m) / n if n > 0 else 0

    return {
        "n": n,
        "m": m,
        "rate": rate,
        "max_col_w": max(col_weights),
        "max_row_w": max(row_weights),
    }


def summarizeAlistFolder(folderPath):
    """
    Scan a folder for .alist files and print summary table.

    Parameters
    ----------
    path : str
        Path to the folder containing ALIST files.

    """
    table = PrettyTable()
    table.field_names = [
        "File",
        "n (length)",
        "m (checks)",
        "Rate",
        "Max Var Deg",
        "Max Check Deg",
    ]

    for filename in os.listdir(folderPath):
        if filename.endswith(".alist") or filename.endswith(".txt"):
            try:
                path = os.path.join(folderPath, filename)
                info = parseAlist(path)
                table.add_row(
                    [
                        filename,
                        info["n"],
                        info["m"],
                        f"{info['rate']:.3f}",
                        info["max_col_w"],
                        info["max_row_w"],
                    ]
                )
            except Exception as e:
                print(f"Failed to parse {filename}: {e}")

    print(table)
