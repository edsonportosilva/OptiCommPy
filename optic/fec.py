# -*- coding: utf-8 -*-

import numpy as np
from commpy.channelcoding.ldpc import triang_ldpc_systematic_encode as enc
from commpy.channelcoding.ldpc import ldpc_bp_decode as dec
import re

def ldpcEncode(b, LDPCparams):
    """
    Encode data bits with binary LDPC code
    b = np.random.randint(2, size=(K, Nwords))

    """
    fecFamily = LDPCparams['filename'][6:11]
    fecID = LDPCparams['filename'][12:]
    
    num = [float(s) for s in re.findall(r'-?\d+\.?\d*', fecID)]    
    n = int(num[0])
       
    if fecFamily == 'AR4JA':
        N = n
    else:
        N = LDPCparams["n_vnodes"]

<<<<<<< Updated upstream
    # generate random interleaver
    interlv = np.random.permutation(N)
=======
    N = n if fecFamily == 'AR4JA' else LDPCparams["n_vnodes"]
>>>>>>> Stashed changes

    # encode bits
    codedBits = enc(b, LDPCparams, pad=False)
    codedBits = (codedBits[0:n, :].T).reshape(1, -1).T

    return  codedBits


def ldpcDecode(llr, deinterlv, LDPCparams, nIter, alg="SPA"):
    """
    Decode binary LDPC encoded data bits
    b = np.random.randint(2, size=(K, Nwords))

    """
    
    fecID = LDPCparams['filename'][12:]
    
    num = [float(s) for s in re.findall(r'-?\d+\.?\d*', fecID)]    
<<<<<<< Updated upstream
    
    N = LDPCparams["n_vnodes"]
    n = int(num[0])
    
=======

    N = LDPCparams["n_vnodes"]   
    n = int(num[0])        
>>>>>>> Stashed changes
    dep = int(N-n)
    
    # generate deinterleaver
<<<<<<< Updated upstream
    deinterlv = interlv.argsort()

    # deinterleave received LLRs
    llr = llr.reshape(-1, n)
    llr = llr[:, deinterlv]
    
    # depuncturing
    if dep > 0:
        llr = np.concatenate((llr, np.zeros((llr.shape[0], dep))), axis=1)
                
=======
    # deinterlv = interlv.argsort()
    
    # reshape received LLRs
    llr = llr.reshape(-1, n)    
        
    # depuncturing
    if dep > 0:
        llr = np.concatenate((llr, np.zeros((llr.shape[0], dep))), axis=1)
    
    decodedBits_hd = (-np.sign(llr)+1)//2 
    decodedBits_hd = decodedBits_hd.reshape(-1, N).T
    
    print(llr.shape)
    
    llr = llr[:, deinterlv]
>>>>>>> Stashed changes
    llr = llr.ravel()
        
    # decode received code words
    decodedBits, llr_out = dec(llr, LDPCparams, alg, nIter)
        
    return decodedBits, decodedBits_hd, llr_out

