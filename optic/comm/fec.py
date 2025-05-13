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
    Hs = np.array(Hs, dtype=np.int8)
    
    # do the necessary column swaps
    cols = np.arange(Hs.shape[1])      
    indP = cols[(np.sum(Hs[:, 0:],axis=0) > 1)]  # indexes of cols belonging to matrix P
    indI = cols[(np.sum(Hs[:, 0:],axis=0) == 1)] # indexes of cols belonging to the indentity
    
    Hnew = np.hstack((Hs[:,cols[indP]], 
                     Hs[:,cols[indI]]))
    
    colSwaps = np.hstack((indP, indI))
        
    # systematic generator matrix G
    G = np.hstack((np.eye(k), Hnew[:,0:k].T))
    
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

def encodeLDPC(H, bits):
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
    G, colSwaps, _ = par2gen(H) # get systematic generator matrix G    
    G = G.astype(np.int32)
    return encoder(G, bits), colSwaps

@njit
def encoder(G, bits):
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
    for col in range(N):  # for each input word
        for i in range(n):  # for each codeword bit
            acc = 0
            for j in range(k):
                acc ^= G[i, j] & bits[j, col]  # binary dot product
            codewords[i, col] = acc
    return codewords

@njit
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
    for var in range(n):
        for check in varNodes[var]:
            msg_v_to_c[check, var] = llr[var]
    
    llr = llr.astype(prec)
    H = H.astype(prec)
    
    for indIter in range(maxIter):
        # Check-to-variable update
        for check in range(m):
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
        for var in range(n):
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
        
        for var in range(n):
            finalLLR[var] = llr[var]
            for check in varNodes[var]:
                finalLLR[var] += msg_c_to_v[check, var]
                decoded_bits[var] = (-np.sign(finalLLR[var]) + 1) // 2                

        if np.all(np.mod(H @ decoded_bits, 2) == 0):
            success = 1
            break            

    return finalLLR.flatten(), indIter, success

def decodeLDPC(llrs, H, maxIter=50, prgsBar=False, prec=np.float64):
    """
    Decodes multiple LDPC codewords using the belief propagation (sum-product) algorithm.

    Parameters
    ----------
    llrs : ndarray of shape (numCodewords, n)
        2D array containing the log-likelihood ratios (LLRs) of each received bit 
        for multiple codewords. Each row corresponds to a different codeword.

    H : scipy.sparse matrix or ndarray of shape (m, n)
        Parity-check matrix of the LDPC code. It defines the structure of the 
        factor graph used in message passing.

    maxIter : int, optional
        Maximum number of belief propagation iterations per codeword (default is 50).

    prgsBar : bool, optional
        If True, displays a progress bar during decoding (default is False).

    prec : data-type, optional
        Data type for the computations (default is np.float64).

    Returns
    -------
    outputLLRs : ndarray of shape (numCodewords, n)
        Array containing the final decoded LLRs for all codewords after belief 
        propagation. Each row corresponds to a decoded codeword.
    """
    H = csr_matrix.todense(H).astype(np.int8)
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
        outputLLRs[indCw, :], indIter, success = sumProductAlgorithm(llrs[indCw, :], H, checkNodes, varNodes, maxIter)
        #outputLLRs[indCw, :] = finalLLR
     
        if success:
            logg.info(f'Frame {indCw} - Successful decoding at iteration {indIter}.')
            continue

    decodedBits = ((-np.sign(outputLLRs)+1)//2).astype(np.int8)
    
    return decodedBits, outputLLRs



