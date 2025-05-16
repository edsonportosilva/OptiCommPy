"""
===============================================================
Forward Error Correction (FEC) utilities (:mod:`optic.comm.fec)
===============================================================

.. autosummary::
   :toctree: generated/

   par2gen                 -- Parity-check matrix to generator matrix conversion
   gaussElim               -- Gaussian elimination over GF(2)
   encoder                 -- Performs linear block encoding
   sumProductAlgorithm     -- Belief propagation decoding using the sum-product algorithm 
   minSumAlgorithm         -- Belief propagation decoding using the min-sum algorithm
   encodeLDPC              -- Encode binary messages using a LDPC parity-check matrix   
   decodeLDPC              -- Decode multiple LDPC codewords using belief propagation    
"""


"""Forward Error Correction (FEC) utilities."""
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from numba import njit, prange
from numba.typed import List
from tqdm import tqdm

import logging as logg

def par2gen(H):
    """
    Convert a binary parity-check matrix H into a systematic generator matrix G over GF(2).

    Parameters
    ----------
    H : ndarray of shape (n - k, n)
        Parity-check matrix with entries in {0, 1}. Typically used in linear block codes such as LDPC codes.

    Returns
    -------
    G : ndarray of shape (k, n)
        Systematic generator matrix corresponding to the input parity-check matrix H.
        The form of G is `[I_k | P]`, where `I_k` is the identity matrix and `P` is a binary matrix.

    colSwaps : ndarray of shape (n,)
        Indices representing the column permutations applied to H to obtain Hnew.
        These permutations are required to match the systematic form used in G.

    Hnew : ndarray of shape (n - k, n)
        The modified parity-check matrix after Gaussian elimination and column reordering.
        This matrix has the identity portion on the right-hand side, corresponding to
        a standard systematic form `[P^T | I_{n-k}]`.

    Notes
    -----
    The function first applies Gaussian elimination over GF(2) to bring H to row echelon form.
    Then, it identifies and reorders columns to isolate an identity submatrix (I) and its
    complement (P). Using the transposed left portion of H, it constructs the corresponding
    systematic generator matrix.

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

    Hs = gaussElim(H) # Reduce matrix to row echelon form
    Hs = np.array(Hs, dtype=np.uint8)
    
    # do the necessary column swaps
    cols = np.arange(Hs.shape[1])      
    indP = cols[(np.sum(Hs[:, 0:],axis=0) > 1)]  # indexes of cols belonging to matrix P
    indI = cols[(np.sum(Hs[:, 0:],axis=0) == 1)] # indexes of cols belonging to the indentity
    
    Hnew = np.hstack((Hs[:,cols[indP]], 
                     Hs[:,cols[indI]]))
    
    colSwaps = np.hstack((indP, indI))
        
    # systematic generator matrix G
    G = np.hstack((np.eye(k,dtype=np.uint8), Hnew[:,0:k].T))
    
    return G, colSwaps, Hnew

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
    Encode binary messages using a parity-check matrix of a linear block code (e.g., LDPC).

    Parameters
    ----------
    bits : ndarray of shape (k, N)
        Binary input messages to be encoded. Each column is a message of length `k`,
        and there are `N` such messages.
    param : object
        Object containing the following attributes:

        - mode : str
            Mode of operation ('DVBS2', 'IEEE_802.11nD2', or 'general').

        - H : ndarray of shape (n - k, n)
            Binary parity-check matrix with entries in {0, 1}. It defines the linear constraints
            of the code and is used to derive a corresponding systematic generator matrix.

        - G : ndarray of shape (k, n), optional
            Generator matrix with entries in {0, 1}. If provided, it will be used for encoding
            instead of deriving it from H.

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
        Binary encoded codewords. Each column is a codeword of length `n` corresponding
        to the respective input message in `bits`. The codewords satisfy `H @ c = 0 (mod 2)`.

    Notes
    -----
    This function performs the following steps:

    1. Uses `par2gen(H)` to compute the corresponding systematic generator matrix `G`
       of the form `[I_k | P]`, and stores the column permutation used to achieve it.
    2. Encodes the input messages using `encoder(G, bits)` to get systematic codewords.
    3. Applies the inverse column permutation to return the codewords in the original bit ordering.

    The arithmetic is performed over GF(2). The result satisfies the parity-check condition
    `H @ codeword = 0 mod 2` for each column.

    """
    # check input parameters
    mode = getattr(param, 'mode', 'DVBS2')
    n = getattr(param, 'n', 64800)
    R = getattr(param, 'R', '4/5')
    H = getattr(param, 'H', None)
    G = getattr(param, 'G', None)
    systematic = getattr(param, 'systematic', True)
    P1 = getattr(param, 'P1', None)
    P2 = getattr(param, 'P2', None)
    path = getattr(param, 'path', None)
    
    if H is None:
        try:            
            filename  = f'\LDPC_{mode}_{n}b_R{R[0]}{R[2]}.txt'
            H = readAlist(path+filename)
            H = csr_matrix.todense(H).astype(np.int8)
            param.H = H
        except FileNotFoundError:
            logg.error(f'File {filename} not found. Please provide a valid parity-check matrix H.')

    if mode == 'DVBS2':
        if H  is None:
            raise ValueError('H is None. Please provide a valid DVBS2 parity-check matrix.')
        return encodeDVBS2(bits, H)
    elif mode == 'IEEE_802.11nD2':
        if P1 is None or P2 is None:
            P1, P2, H = triangP1P2(H)
            param.P1 = P1
            param.P2 = P2
            param.H = H
        return encodeTriang(bits, P1, P2)
    elif mode == 'AR4JA':
        if G is None:
            G, _, Hnew = par2gen(H) # get systematic generator matrix G 
            G = G.astype(np.uint8)        
            #G = G[:,0:n]        
            param.G = G
            param.H = Hnew#[:,0:n]      
            return encoder(G, bits, systematic)
        else:
            G = G.astype(np.uint8)
            return encoder(G, bits, systematic)
    else:       
        logg.error(f'Unsupported mode: {mode}. Supported modes are: DVBS2, IEEE_802.11nD2, AR4JA.')

    
@njit(parallel=True)
def encodeDVBS2(bits, H):
    """
    Encode multiple binary messages using a DVB-S2 LDPC parity-check matrix.

    Parameters
    ----------
    bits : ndarray of shape (k, N)
        Binary input messages to be encoded. Each column represents a message of length `k`,
        and there are `N` such messages.

    H : ndarray of shape (n - k, n)
        Binary parity-check matrix with entries in {0, 1}. It defines the linear constraints
        of the code and is used to derive a corresponding systematic generator matrix.

    Returns
    -------
    codewords : ndarray of shape (n, N)
        Binary encoded codewords. Each column is a codeword of length `n` corresponding
        to the respective input message in `bits`. The codewords satisfy H @ c = 0 mod 2.
    """
    H = H.astype(np.uint8)
    bits = bits.astype(np.uint8)

    m, n = H.shape
    k = n - m
    N = bits.shape[1]
    A = H[:, :k]  # shape (m, k)

    codewords = np.zeros((n, N), dtype=np.uint8)

    for col in prange(N):
        # Copy message bits
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
    Encode binary messages using a generator matrix over GF(2).

    Parameters
    ----------
    G : ndarray of shape (k, n)
        Generator matrix with entries in {0, 1}. Each row corresponds to a basis vector
        of the code, and the matrix defines the linear transformation from input bits to codewords.

    bits : ndarray of shape (k, N)
        Binary input messages to encode. Each column is a message of length `k`, and there are `N` messages
        in total.

    systematic : bool, optional
        If True, the generator matrix is assumed to be in systematic form. If False, the
        generator matrix is treated as a general linear transformation (default is True).

    Returns
    -------
    codewords : ndarray of shape (n, N)
        Encoded binary codewords. Each column corresponds to a codeword of length `n` resulting
        from applying the generator matrix to the respective input message.

    Notes
    -----
    This function performs matrix multiplication over GF(2), i.e., using XOR for addition
    and AND for multiplication.

    The function transposes the generator matrix internally so that each codeword is computed as:

        codeword = G.T @ bits[:, i] mod 2

    which is equivalent to bits[i] @ G mod 2 in standard linear coding.

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


@njit(parallel=True)
def sumProductAlgorithm(llr, H, checkNodes, varNodes, maxIter, prec=np.float64):
    """
    Performs belief propagation decoding using the sum-product algorithm
    for a single LDPC codeword.

    Parameters
    ----------
    llr : ndarray of shape (n,)
        Array of log-likelihood ratios (LLRs) for each bit of the received codeword.

    H : ndarray of shape (m, n)
        Binary parity-check matrix of the LDPC code. It is used to enforce parity constraints.

    checkNodes : list of ndarray
        List of length `m`, where each element is a 1D array containing the indices
        of variable nodes (bits) involved in the corresponding check node (parity-check equation).

    varNodes : list of ndarray
        List of length `n`, where each element is a 1D array containing the indices
        of check nodes that the corresponding variable node participates in.

    maxIter : int
        Maximum number of belief propagation iterations.

    prec : data-type, optional
        Data type for the computations (default is np.float64).

    Returns
    -------
    finalLLR : ndarray of shape (n,)
        Updated log-likelihood ratios after message passing.

    numIter : int
        Number of iterations executed until decoding converged or reached `maxIter`.

    success : int
        Indicates whether decoding was successful (1) or not (0),
        based on the parity-check condition.
    """
    m, n = H.shape
    msg_v_to_c = np.zeros((m, n), dtype=prec) 
    msg_c_to_v = np.zeros((m, n), dtype=prec)
    success = False    

    # Initialize variable-to-check messages with input LLRs
    for var in prange(n):
        for check in varNodes[var]:
            msg_v_to_c[check, var] = llr[var]
    
    llr = llr.astype(prec)
    H = H.astype(prec)
    
    for indIter in range(maxIter):
        # Check-to-variable update
        for check in prange(m):
            for var_idx in range(len(checkNodes[check])):
                var = checkNodes[check][var_idx]
                product = 1.0
                for neighbor_idx in range(len(checkNodes[check])):
                    neighbor = checkNodes[check][neighbor_idx]
                    if neighbor != var:
                        product *= np.tanh(msg_v_to_c[check, neighbor] / 2)
                product = min(0.999999, max(-0.999999, product))  # clip
                msg_c_to_v[check, var] = 2 * np.arctanh(product)
        
        # Variable-to-check update
        for var in prange(n):
            for check_idx in range(len(varNodes[var])):
                check = varNodes[var][check_idx]
                sum_msg = llr[var]
                for neighbor_idx in range(len(varNodes[var])):
                    neighbor = varNodes[var][neighbor_idx]
                    if neighbor != check:
                        sum_msg += msg_c_to_v[neighbor, var]
                msg_v_to_c[check, var] = sum_msg

        # Final LLR computation
        finalLLR = np.zeros((n, 1), dtype=prec)
        decoded_bits = np.zeros((n, 1), dtype=prec)
        
        for var in prange(n):
            finalLLR[var] = llr[var]
            for check in varNodes[var]:
                finalLLR[var] += msg_c_to_v[check, var]
                decoded_bits[var] = (-np.sign(finalLLR[var]) + 1) // 2                

        if np.all(np.mod(H @ decoded_bits, 2) == 0):
            success = True
            break            

    return finalLLR.flatten(), indIter, success

@njit(parallel=True)
def minSumAlgorithm(llr, H, checkNodes, varNodes, maxIter, prec=np.float64):
    """
    Performs LDPC decoding using the Min-Sum Algorithm for a single codeword.

    This function implements the belief propagation decoding using the Min-Sum 
    approximation of the Sum-Product Algorithm (SPA). It replaces the computationally
    expensive tanh and arctanh operations with simple min and sign operations at
    check nodes, making it more suitable for hardware or parallel implementations.

    Parameters
    ----------
    llr : ndarray of shape (n,)
        Log-likelihood ratios (LLRs) of the received codeword bits.

    H : ndarray of shape (m, n)
        Binary parity-check matrix representing the LDPC code.

    checkNodes : list of ndarray
        List of length `m`, where each entry contains the indices of variable nodes 
        connected to the corresponding check node.

    varNodes : list of ndarray
        List of length `n`, where each entry contains the indices of check nodes 
        connected to the corresponding variable node.

    maxIter : int
        Maximum number of iterations for belief propagation.

    prec : data-type, optional
        Numerical precision to use in computations (default is np.float64).

    Returns
    -------
    finalLLR : ndarray of shape (n,)
        Updated LLR values for the decoded codeword after the final iteration.

    numIter : int
        Number of iterations performed before successful decoding or reaching `maxIter`.

    success : int
        1 if decoding succeeded (i.e., all parity-check equations are satisfied), 
        0 otherwise.
    """
    m, n = H.shape
    msg_v_to_c = np.zeros((m, n), dtype=prec)
    msg_c_to_v = np.zeros((m, n), dtype=prec)
    success = 0

    # Initialize variable-to-check messages with input LLRs
    for var in prange(n):
        for check in varNodes[var]:
            msg_v_to_c[check, var] = llr[var]

    llr = llr.astype(prec)
    H = H.astype(prec)

    for indIter in range(maxIter):
        # Check-to-variable update (Min-Sum)
        for check in prange(m):
            for var in checkNodes[check]:
                sign_product = 1
                min_abs = np.inf
                for neighbor in checkNodes[check]:
                    if neighbor != var:
                        val = msg_v_to_c[check, neighbor]
                        sign_product *= np.sign(val)
                        min_abs = min(min_abs, abs(val))
                msg_c_to_v[check, var] = sign_product * min_abs

        # Variable-to-check update
        for var in prange(n):
            for check in varNodes[var]:
                sum_msg = llr[var]
                for neighbor in varNodes[var]:
                    if neighbor != check:
                        sum_msg += msg_c_to_v[neighbor, var]
                msg_v_to_c[check, var] = sum_msg

        # Final LLR and decision
        finalLLR = np.zeros((n, 1), dtype=prec)
        decoded_bits = np.zeros((n, 1), dtype=prec)

        for var in prange(n):
            finalLLR[var] = llr[var]
            for check in varNodes[var]:
                finalLLR[var] += msg_c_to_v[check, var]
            decoded_bits[var] = (-np.sign(finalLLR[var]) + 1) // 2

        if np.all(np.mod(H @ decoded_bits, 2) == 0):
            success = 1
            break

    return finalLLR.flatten(), indIter, success


def decodeLDPC(llrs, param):
    """
    Decodes multiple LDPC codewords using the belief propagation (sum-product) algorithm.

    Parameters
    ----------
    llrs : ndarray of shape (numCodewords, n)
        Array of log-likelihood ratios (LLRs) for each bit of the received codewords.
        Each row corresponds to a codeword, and each column corresponds to a bit.
        
    param : object
        Object containing the following attributes:

        - H : ndarray of shape (m, n)
            Binary parity-check matrix of the LDPC code.

        - maxIter : int
            Maximum number of iterations for belief propagation.

        - alg : str
            Decoding algorithm to use ('SPA' or 'MSA').

        - prgsBar : bool
            If True, displays a progress bar during decoding.

        - prec : data-type
            Numerical precision to use in computations (default is np.float64). 

    Returns
    -------
    decodedBits : ndarray of shape (numCodewords, n)
        Array of decoded bits for each codeword. Each row corresponds to a codeword,
        and each column corresponds to a bit.

    outputLLRs : ndarray of shape (numCodewords, n)
        Array of updated log-likelihood ratios (LLRs) after decoding. Each row corresponds
        to a codeword, and each column corresponds to a bit.
    """
    # check input parameters
    H = getattr(param, 'H', None)
    maxIter = getattr(param, 'maxIter', 25)
    alg = getattr(param, 'alg', 'SPA')
    prgsBar = getattr(param, 'prgsBar', False)
    prec = getattr(param, 'prec', np.float64)

    if H is None:
        logg.error('H is None. Please provide a valid parity-check matrix.')

    if type(H) == csr_matrix:
        H = csr_matrix.todense(H).astype(np.uint8) 
    elif type(H) == csc_matrix:
        H = csc_matrix.todense(H).astype(np.uint8) 
    elif type(H) == coo_matrix:
        H = coo_matrix.todense(H).astype(np.uint8)
    else:
        H = H.astype(np.uint8)

    m, n = H.shape
    numCodewords = llrs.shape[0]

    llrs = np.clip(llrs, -200, 200)
    outputLLRs = np.zeros_like(llrs, dtype=prec)

    # Build adjacency lists using fixed-size lists for Numba       
    checkNodes = List([np.where(H[i, :] == 1)[1].astype(np.int32) for i in range(m)])
    varNodes = List([np.where(H[:, j] == 1)[0].astype(np.int32) for j in range(n)])

    # Convert H to binary array
    H = np.array(H, dtype=np.int8)
    logg.info( f'LDPC decoding: {numCodewords} codewords')
    for indCw in tqdm(range(numCodewords), disable=not (prgsBar)):  
        if alg == 'SPA':      
            outputLLRs[indCw, :], indIter, success = sumProductAlgorithm(llrs[indCw, :], H, checkNodes, varNodes, maxIter)
        elif alg == 'MSA':
            outputLLRs[indCw, :], indIter, success = minSumAlgorithm(llrs[indCw, :], H, checkNodes, varNodes, maxIter)
             
        if success:
            logg.info(f'Frame {indCw} - Successful decoding at iteration {indIter}.')
            continue

    decodedBits = ((-np.sign(outputLLRs)+1)//2).astype(np.int8)
    
    return decodedBits, outputLLRs


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
    var_degrees = [int(np.sum(H[:, j])) for j in range(n)]
    check_degrees = [int(np.sum(H[i, :])) for i in range(m)]
    max_col_deg = max(var_degrees)
    max_row_deg = max(check_degrees)

    with open(filename, 'w') as f:
        f.write(f"{n} {m}\n")
        f.write(f"{max_col_deg} {max_row_deg}\n")

        f.write(' '.join(str(d) for d in var_degrees) + '\n')
        f.write(' '.join(str(d) for d in check_degrees) + '\n')

        # Variable node connections (1-based indexing)
        for j in range(n):
            connections = np.where(H[:, j]==1)[0] + 1
            padded = list(connections) + [0] * (max_col_deg - len(connections))
            f.write(' '.join(str(i) for i in padded) + '\n')

        # Check node connections (1-based indexing)
        for i in range(m):
            connections = np.where(H[i, :]==1)[1] + 1
            padded = list(connections) + [0] * (max_row_deg - len(connections))
            f.write(' '.join(str(j) for j in padded) + '\n')
    
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
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n, m = map(int, lines[0].split())  
    var_conn_lines = lines[4:4 + n]
    
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
        Binary matrix with entries in {0, 1}.

    Returns
    -------
    A_inv : ndarray of shape (n, n), dtype=np.uint8
        Inverse of A over GF(2), if invertible. If not invertible, returns the identity matrix.

    success : bool
        True if A is invertible, False otherwise.
    """
    n = A.shape[0]
    A = A.copy()
    A_inv = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        A_inv[i, i] = 1  # identity matrix

    for i in range(n):
        # Find pivot
        if A[i, i] == 0:
            found = False
            for j in range(i + 1, n):
                if A[j, i] == 1:
                    # Manually swap rows i and j in A and A_inv
                    for k in range(n):
                        tmp = A[i, k]
                        A[i, k] = A[j, k]
                        A[j, k] = tmp
                        tmp = A_inv[i, k]
                        A_inv[i, k] = A_inv[j, k]
                        A_inv[j, k] = tmp
                    found = True
                    break
            if not found:
                return A_inv, False  # Matrix is not invertible

        # Eliminate all other entries in column i
        for j in range(n):
            if j != i and A[j, i] == 1:
                for k in range(n):
                    A[j, k] ^= A[i, k]
                    A_inv[j, k] ^= A_inv[i, k]

    return A_inv, True

@njit
def triangularize(H):
    """
    Convert binary matrix H into lower-triangular form using only row and column permutations.
    Numba-compatible version.

    Parameters
    ----------
    H : ndarray of shape (m, n), dtype=np.uint8
        Binary parity-check matrix with entries in {0, 1}.

    Returns
    -------
    H_tri : ndarray of shape (m, n), dtype=np.uint8
        Triangularized matrix.

    row_perm : ndarray of shape (m,), dtype=np.int32
        Row permutation indices.

    col_perm : ndarray of shape (n,), dtype=np.int32
        Column permutation indices.
    """
    m, n = H.shape
    H_tri = H.copy()
    row_perm = np.arange(m, dtype=np.int32)
    col_perm = np.arange(n, dtype=np.int32)

    for i in range(m):
        pivot_found = False

        for r in range(i, m):
            for c in range(i, n):
                if H_tri[r, c] == 1:
                    # Swap rows
                    if r != i:
                        for j in range(n):
                            H_tri[i, j], H_tri[r, j] = H_tri[r, j], H_tri[i, j]
                        row_perm[i], row_perm[r] = row_perm[r], row_perm[i]
                    # Swap columns
                    if c != i:
                        for j in range(m):
                            H_tri[j, i], H_tri[j, c] = H_tri[j, c], H_tri[j, i]
                        col_perm[i], col_perm[c] = col_perm[c], col_perm[i]
                    pivot_found = True
                    break
            if pivot_found:
                break

        if not pivot_found:
            # Leave zeros if no pivot found in this column
            continue

        # Eliminate below
        for r in range(i + 1, m):
            if H_tri[r, i] == 1:
                for j in range(n):
                    H_tri[r, j] ^= H_tri[i, j]

    return H_tri, row_perm, col_perm

def triangP1P2(H):
    """
    Convert a binary parity-check matrix H into a lower-triangular form and extract matrices P1 and P2.

    Parameters
    ----------
    H : ndarray of shape (m, n)
        Binary parity-check matrix with entries in {0, 1}.
        It is used to derive the matrices P1 and P2.
    Returns
    -------
    P1 : ndarray of shape (m1, k)
        First parity matrix.
    P2 : ndarray of shape (m2, k)
        Second parity matrix.
    H_tri : ndarray of shape (m, n)
        Triangularized H matrix.
    """
    H = H.astype(np.uint8)

    # convert to lower-triangular form
    H_tri, _, _ = triangularize(H)

    # calculate the gap g
    idx = np.where(H_tri[:,-1]==1)
    g = H_tri.shape[0] - np.min(idx[0])-1
    m = H_tri.shape[0]
    n = H_tri.shape[1]
    k = n - m

    # extract matrices
    E = H_tri[m-g:, n-(m-g):]
    T = H_tri[0:m-g, n-(m-g):]
    A = H_tri[0:m-g, 0:k]
    B = H_tri[0:m-g, k:k+g]
    C = H_tri[m-g:, 0:k]
    D = H_tri[m-g:, k:k+g]

    # invert matrix T
    T_inv, found = inverseMatrixGF2(T)
    if not found:
        logg.error('Matrix T is not invertible.')

    X = np.mod(E@T_inv, 2)           
    C_tilde = np.mod( X @ A + C, 2)
    D_tilde = np.mod( X @ B + D, 2)

    # invert matrix D tilde
    D_tilde_inv, found = inverseMatrixGF2(D_tilde)
    if not found:
        logg.error('Matrix D_tilde is not invertible.')       

    P1 = np.mod(D_tilde_inv@C_tilde, 2) 
    P2 = np.mod(T_inv @ np.mod(A + np.mod(B @ P1, 2), 2), 2)

    return P1, P2, H_tri
        

@njit(parallel=True)
def encodeTriang(bits, P1, P2):
    """
    Encode binary messages using two parity matrices for LDPC encoding.

    Parameters
    ----------
    bits : ndarray of shape (k, N)
        Binary input messages. Each column is a message to be encoded.

    P1 : ndarray of shape (m1, k)
        First parity matrix.

    P2 : ndarray of shape (m2, k)
        Second parity matrix.

    Returns
    -------
    codewords : ndarray of shape (k + m1 + m2, N)
        Encoded codewords, one per column.
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
