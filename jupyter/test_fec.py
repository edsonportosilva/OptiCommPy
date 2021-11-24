# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from commpy.modulation import QAMModem, PSKModem
from optic.metrics import signal_power, calcLLR, fastBERcalc
from optic.fec import ldpcEncode, ldpcDecode
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import io
from tqdm.notebook import tqdm
from numba import njit

import os.path as path
# -

from commpy.channelcoding import ldpc
from commpy.channelcoding.ldpc import triang_ldpc_systematic_encode as encodeLDPC
from commpy.channelcoding.ldpc import ldpc_bp_decode as decodeLDPC
from commpy.channelcoding.interleavers import RandInterlv


# %load_ext autoreload
# %autoreload 2

@njit
def awgn(tx, noiseVar):
    
    σ        = np.sqrt(noiseVar)
    noise    = np.random.normal(0,σ, tx.size) + 1j*np.random.normal(0,σ, tx.size)
    noise    = 1/np.sqrt(2)*noise
    
    rx = tx + noise
    
    return rx


# ## Create LDPCparam files

# +
# path = r'C:\Users\edson\Documents\GitHub\edsonportosilva\robochameleon-private\addons\AR4JA_LDPC_FEC'

# d = sp.io.loadmat(path+'\LDPC_AR4JA_6144b_R23.mat')
# H = d['H']

# # H = d['LDPC']['H'] # parity check matrix
# # H = H[0][0][0][0][0]
# H = sp.sparse.csr_matrix.todense(H).astype(np.int8)
# H = np.asarray(H)

# file_path = r'C:\Users\edson\Documents\GitHub\edsonportosilva\OpticCommPy\optic\fecParams\LDPC_AR4JA_6144b_R23.txt'

# ldpc.write_ldpc_params(H, file_path)

# +
# FEC parameters
family = "11nD2"
R = 56
n = 648

mainDir  = path.abspath(path.join("../")) 
filename = '\LDPC_' + family + '_' + str(n) + 'b_R' + str(R) + '.txt'
filePath = mainDir + r'\optic\fecParams' + filename
filePath

# +
# Run AWGN simulation 
EbN0dB = 16
M      = 256
Nwords = 800
nIter  = 20

# FEC parameters
LDPCparams = ldpc.get_ldpc_code_params(filePath)
K = LDPCparams['n_vnodes'] - LDPCparams['n_cnodes']

# modulation parameters
mod = QAMModem(m=M)
constSymb = mod.constellation
bitMap = mod.demodulate(constSymb, demod_type="hard")
bitMap = bitMap.reshape(-1, int(np.log2(M)))
Es = mod.Es

# generate random bits
bits = np.random.randint(2, size = (K, Nwords))

# encode data bits with LDPC soft-FEC
bitsTx, codedBitsTx, interlv = ldpcEncode(bits, LDPCparams)

# Map bits to constellation symbols
symbTx = mod.modulate(bitsTx)

# Normalize symbols energy to 1
symbTx = symbTx/np.sqrt(Es)

# AWGN    
snrdB    = EbN0dB + 10*np.log10(np.log2(M))
noiseVar = 1/(10**(snrdB/10))

symbRx = awgn(symbTx, noiseVar)

# pre-FEC BER calculation (hard demodulation)
BER, _, _ = fastBERcalc(symbRx, symbTx, mod)
print('BER = %.2e'%BER[0])

# soft-demodulation
llr = calcLLR(symbRx, noiseVar, constSymb/np.sqrt(Es), bitMap)

# soft-FEC decoding
decodedBits, llr_out = ldpcDecode(llr, interlv, LDPCparams, nIter, alg="SPA")

# post-FEC BER calculation
BERpost = np.mean(np.logical_xor(codedBitsTx, decodedBits))

print('BERpostFEC = %.2e'%BERpost)
print('Number of bits = ', decodedBits.size)
# -



# +
Nwords = 800
nIter  = 20

# FEC parameters
LDPCparams = ldpc.get_ldpc_code_params(filePath)
K = LDPCparams['n_vnodes'] - LDPCparams['n_cnodes']

# Run BER vs Ebn0 Monte Carlo simulation 
qamOrder  = [16]  # Modulation order
EbN0dB_  = np.arange(2, 8, 0.125)

BERpre   = np.zeros((len(EbN0dB_),len(qamOrder)))
BERpost  = np.zeros((len(EbN0dB_),len(qamOrder)))

BERpre[:]  = np.nan
BERpost[:] = np.nan

for ii, M in enumerate(qamOrder):
    print('run sim: M = ', M)
    
    # modulation parameters
    mod = QAMModem(m=M)
    constSymb = mod.constellation
    bitMap = mod.demodulate(constSymb, demod_type="hard")
    bitMap = bitMap.reshape(-1, int(np.log2(M)))
    Es = mod.Es

    for indSNR in tqdm(range(EbN0dB_.size)):
        
        EbN0dB = EbN0dB_[indSNR]
        
        # generate random bits
        bits = np.random.randint(2, size = (K, Nwords))

        # encode data bits with LDPC soft-FEC
        bitsTx, codedBitsTx, interlv = ldpcEncode(bits, LDPCparams)

        # Map bits to constellation symbols
        symbTx = mod.modulate(bitsTx)

        # Normalize symbols energy to 1
        symbTx = symbTx/np.sqrt(Es)

        # AWGN    
        snrdB    = EbN0dB + 10*np.log10(np.log2(M))
        noiseVar = 1/(10**(snrdB/10))

        symbRx = awgn(symbTx, noiseVar)

        # pre-FEC BER calculation (hard demodulation)
        BERpre[indSNR, ii], _, _ = fastBERcalc(symbRx, symbTx, mod)
        #print('BER = %.2e'%BERpre[indSNR, ii])

        # soft-demodulation
        llr = calcLLR(symbRx, noiseVar, constSymb/np.sqrt(Es), bitMap)

        # soft-FEC decoding
        decodedBits, llr_out = ldpcDecode(llr, interlv, LDPCparams, nIter, alg="SPA")

        # post-FEC BER calculation
        BERpost[indSNR, ii] = np.mean(np.logical_xor(codedBitsTx, decodedBits))
        #print('BERpostFEC = %.2e'%BERpost[indSNR, ii])

# +
# Plot simulation results and theoretical curves        
BERpre[BERpre==0] = np.nan
BERpost[BERpost==0] = np.nan

plt.figure(figsize=(10,6))
for ii, M in enumerate(qamOrder):
    plt.plot(EbN0dB_, np.log10(BERpre[:,ii]),'o-', label=str(M)+'QAM monte carlo [pre]')

#plt.gca().set_prop_cycle(None)

for ii, M in enumerate(qamOrder):
    plt.plot(EbN0dB_, np.log10(BERpost[:,ii]),'x-', label=str(M)+'QAM monte carlo [post]')

plt.xlim(min(EbN0dB_), max(EbN0dB_))
plt.ylim(-6, 0)
plt.legend();
plt.xlabel('EbN0 [dB]');
plt.ylabel('log10(BER)');
plt.grid()
# -


