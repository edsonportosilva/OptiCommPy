# -*- coding: utf-8 -*-

import re

import numpy as np
from commpy.channelcoding.ldpc import ldpc_bp_decode as dec
from commpy.channelcoding.ldpc import triang_ldpc_systematic_encode as enc


def ldpcEncode(b, LDPCparams):
    """
    Encode data bits with binary LDPC code
    b = np.random.randint(2, size=(K, Nwords))

    """
    fecFamily = LDPCparams['filename'][6:11]
    fecID = LDPCparams['filename'][12:]

    num = [float(s) for s in re.findall(r'-?\d+\.?\d*', fecID)]
    n = int(num[0])

    N = n if fecFamily == 'AR4JA' else LDPCparams["n_vnodes"]
    # generate random interleaver
    interlv = np.random.permutation(N)

    # encode bits
    codedBits = enc(b, LDPCparams, pad=False)
    interCodedBits = (codedBits[interlv, :].T).reshape(1, -1).T

    return interCodedBits, codedBits, interlv


def ldpcDecode(llr, interlv, LDPCparams, nIter, alg="SPA"):
    """
    Decode binary LDPC encoded data bits
    b = np.random.randint(2, size=(K, Nwords))

    """

    fecID = LDPCparams['filename'][12:]

    num = [float(s) for s in re.findall(r'-?\d+\.?\d*', fecID)]    

    N = LDPCparams["n_vnodes"]
    n = int(num[0])

    dep = int(N-n)

    # generate deinterleaver
    deinterlv = interlv.argsort()

    # deinterleave received LLRs
    llr = llr.reshape(-1, n)
    llr = llr[:, deinterlv]

    # depuncturing
    if dep > 0:
        llr = np.concatenate((llr, np.zeros((llr.shape[0], dep))), axis=1)

    llr = llr.ravel()

    # decode received code words
    decodedBits, llr_out = dec(llr, LDPCparams, alg, nIter)

    return decodedBits, llr_out

