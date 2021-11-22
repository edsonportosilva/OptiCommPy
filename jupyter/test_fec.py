# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Test-LDPC-decoding" data-toc-modified-id="Test-LDPC-decoding-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Test LDPC decoding</a></span></li></ul></div>
# -

from commpy.modulation import QAMModem, PSKModem
from optic.metrics import signal_power, calcLLR, fastBERcalc
from optic.fec import loggaldecode
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from tqdm.notebook import tqdm
from numba import njit
from numba.typed import List


# +
@njit
def awgn(tx, noiseVar):
    
    σ        = np.sqrt(noiseVar)
    noise    = np.random.normal(0,σ, tx.size) + 1j*np.random.normal(0,σ, tx.size)
    noise    = 1/np.sqrt(2)*noise
    
    rx = tx + noise
    
    return rx

def sparse(H):
    
    M, N = H.shape

    Nl = []
    Ml = []
    for m in range(M):
        Nl.append([])

    for n in range(N):
        Ml.append([])

    # Build the sparse representation of A using the M and N sets

    for m in range(M):
        for n in range(N):
            if H[m, n]:
                Nl[m].append(n)
                Ml[n].append(m)
    
    return List(Nl), List(Ml)


# -

# ## Test LDPC decoding

# +
# A used in chapter
A = np.array([[1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
              [1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
              [0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
              [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
              [1, 1, 0, 1, 0, 0, 1, 1, 1, 0]])

# Inverse of first part, to get systematic form (not needed for decoding)
Apinv = np.array([[1, 0, 1, 1, 0],
                  [0, 1, 1, 0, 1],
                  [0, 1, 0, 1, 1],
                  [1, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1]])

H = (Apinv@A)%2   # systematic parity check matrix

M, N = A.shape
K = N-M;
P = H[:,N-K:N];
G = np.concatenate((P, np.eye(int(K)))) # now A*G = 0 (mod 2)

m = np.array([[1, 0, 1, 0, 1]]).T
c = (G@m)%2;
t = 2*(2*c-1);

a = 2;   # signal amplitude
sigma2 = 2;  # noise variance

# First set the channel posterior probabilities
p1 = np.array([[.22, .16,  .19,  .48, .55,  .87, .18, .79, .25, .76]]).T

# then compute the received values that correspond to these
r =  np.log((1./p1)-1)/(-2*a)*sigma2;  # received vector

x0 = p1 > 0.5
z0 = (A@x0)%2

Nloop = 50;

Lc = 2*a/sigma2;
Nl, Ml = sparse(H)

#x = galdecode(A,p1,Nloop)
lamb, x = loggaldecode(H, r, Nloop,Lc, Nl, Ml)

# +
# Run AWGN simulation 
EbN0dB = 12
M      = 16
Nwords = 10000
Nloop  = 50
Lc = 1

# generate random bits
bitsTx = np.random.randint(2, size=(5, Nwords)) 
encodedBitsTx = ((G@bitsTx)%2).astype('int')
encodedBitsTx = (encodedBitsTx.T).reshape(1,-1).T

# Map bits to constellation symbols
mod = QAMModem(m=M)
symbTx = mod.modulate(encodedBitsTx)

# Normalize symbols energy to 1
symbTx = symbTx/np.sqrt(mod.Es)

# AWGN    
snrdB    = EbN0dB + 10*np.log10(np.log2(M))
noiseVar = 1/(10**(snrdB/10))

symbRx = awgn(symbTx, noiseVar)

# BER calculation
BER, _, _ = fastBERcalc(symbRx, symbTx, mod)
print('BER = ', BER[0])

constSymb = mod.constellation
bitMap = mod.demodulate(constSymb, demod_type="hard")
bitMap = bitMap.reshape(-1, int(np.log2(M)))
Es = mod.Es

llr = calcLLR(symbRx, noiseVar, constSymb / np.sqrt(Es), bitMap)
llr = llr.reshape(-1,1)

decBits = np.zeros((10*Nwords,1))
decBits[:] = np.nan

for k in range(Nwords):  
    llr_in = llr[10*k:10*k+10, :]
    llr_out,_ = loggaldecode(H, llr_in, Nloop, Lc, Nl, Ml)
    decBits[10*k:10*k+10,:] = ( np.sign(-llr_out) + 1 )/2

BERpost = np.mean(np.logical_xor(encodedBitsTx, decBits))

print('BERpostFEC = ', BERpost)
# -

