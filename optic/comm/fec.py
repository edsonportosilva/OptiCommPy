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
def gaussElim(matrix):
    """
    Perform Gaussian elimination over GF(2) to reduce a binary matrix to row echelon form.

    Parameters
    ----------
    matrix : ndarray of uint8
        Input binary matrix (dense, 2D). All operations are over GF(2).

    Returns
    -------
    matrix : ndarray of uint8
        Matrix in row echelon form (mod 2).
    """
    rowCount, columnCount = matrix.shape
    lead = 0

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

def encodeLDPC(H, bits, G=None, systematic=True):
    """
    Encode binary messages using a parity-check matrix of a linear block code (e.g., LDPC).

    Parameters
    ----------
    H : ndarray of shape (n - k, n)
        Binary parity-check matrix with entries in {0, 1}. It defines the linear constraints
        of the code and is used to derive a corresponding systematic generator matrix.

    bits : ndarray of shape (k, N)
        Binary input messages to be encoded. Each column represents a message of length `k`,
        and there are `N` such messages.

    G : ndarray of shape (k, n), optional
        Systematic generator matrix. If not provided, it will be computed from `H` using `par2gen()`.

    systematic : bool, optional
        If True, the generator matrix is assumed to be in systematic form. If False, the
        generator matrix is treated as a general linear transformation (default is True).

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
    if G is None:
        G, colSwaps, _ = par2gen(H) # get systematic generator matrix G 
        G = G.astype(np.uint8)
        return encoder(G, bits, systematic), colSwaps
    else:   
        G = G.astype(np.uint8)
        return encoder(G, bits, systematic)
    

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
    success = 0    

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
            success = 1
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
    maxIter = getattr(param, 'maxIter', 50)
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
