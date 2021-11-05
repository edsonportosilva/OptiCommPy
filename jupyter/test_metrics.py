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

# + [markdown] toc=true
# # Test transmission performance metrics for the AWGN channel
# -

from commpy.modulation import QAMModem, PSKModem
from optic.metrics import signal_power, monteCarloGMI, monteCarloMI, fastBERcalc, theoryBER
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from tqdm.notebook import tqdm
from numba import njit


@njit
def awgn(tx, noiseVar):
    
    σ        = np.sqrt(noiseVar)
    noise    = np.random.normal(0,σ, tx.size) + 1j*np.random.normal(0,σ, tx.size)
    noise    = 1/np.sqrt(2)*noise
    
    rx = tx + noise
    
    return rx


# ## Test bit-error-rate (BER) versus signal-to-noise ratio per bit ($E_b/N_0$)

# ### QAM constellations with Gray mapping

# +
# Run BER vs Ebn0 Monte Carlo simulation 

qamOrder  = [4, 16, 64, 256, 1024, 4096]  # Modulation order

EbN0dB_  = np.arange(-2, 35, 1)
BER      = np.zeros((len(EbN0dB_),len(qamOrder)))
BER[:]   = np.nan

for ii, M in enumerate(qamOrder):
    print('run sim: M = ', M)
    for indSNR in tqdm(range(EbN0dB_.size)):

        EbN0dB = EbN0dB_[indSNR]

        # generate random bits
        bitsTx = np.random.randint(2, size=2**19)    

        # Map bits to constellation symbols
        mod = QAMModem(m=M)
        symbTx = mod.modulate(bitsTx)

        # Normalize symbols energy to 1
        symbTx = symbTx/np.sqrt(mod.Es)

        # AWGN    
        snrdB    = EbN0dB + 10*np.log10(np.log2(M))
        noiseVar = 1/(10**(snrdB/10))

        symbRx = awgn(symbTx, noiseVar)

        # BER calculation
        BER[indSNR, ii], _, _ = fastBERcalc(symbRx, symbTx, mod)
        
        if BER[indSNR, ii] == 0:              
            break


# +
# Plot simulation results and theoretical curves        
BER[BER==0] = np.nan

plt.figure(figsize=(10,6))
for ii, M in enumerate(qamOrder):
    plt.plot(EbN0dB_, np.log10(BER[:,ii]),'o', label=str(M)+'QAM monte carlo')

plt.gca().set_prop_cycle(None)

for M in qamOrder:
    BERtheory = theoryBER(M, EbN0dB_,'qam')
    BERtheory[BERtheory==0] = np.nan
    plt.plot(EbN0dB_, np.log10(BERtheory),'-', label=str(M)+'QAM theory')

plt.xlim(min(EbN0dB_), max(EbN0dB_))
plt.ylim(-6, 0)
plt.legend();
plt.xlabel('EbN0 [dB]');
plt.ylabel('log10(BER)');
plt.grid()
# -

# ### PSK constellations with Gray mapping

# +
# Run BER vs Ebn0 Monte Carlo simulation 

pskOrder  = [4, 8, 16, 64]  # Modulation order

EbN0dB_  = np.arange(-2, 35, 1)
BER      = np.zeros((len(EbN0dB_),len(pskOrder)))
BER[:]   = np.nan

for ii, M in enumerate(pskOrder):
    print('run sim: M = ', M)
    for indSNR in tqdm(range(EbN0dB_.size)):

        EbN0dB = EbN0dB_[indSNR]

        # generate random bits
        bitsTx = np.random.randint(2, size=2**19)    

        # Map bits to constellation symbols
        mod = PSKModem(m=M)
        symbTx = mod.modulate(bitsTx)

        # Normalize symbols energy to 1
        symbTx = symbTx/np.sqrt(mod.Es)

        # AWGN    
        snrdB    = EbN0dB + 10*np.log10(np.log2(M))
        noiseVar = 1/(10**(snrdB/10))

        symbRx = awgn(symbTx, noiseVar)

        # BER calculation
        BER[indSNR, ii], _, _ = fastBERcalc(symbRx, symbTx, mod)
        
        if BER[indSNR, ii] == 0:              
            break

# +
# Plot simulation results and theoretical curves        
BER[BER==0] = np.nan

plt.figure(figsize=(10,6))
for ii, M in enumerate(pskOrder):
    plt.plot(EbN0dB_, np.log10(BER[:,ii]),'o', label=str(M)+'PSK monte carlo')

plt.gca().set_prop_cycle(None)

for M in pskOrder:
    BERtheory = theoryBER(M, EbN0dB_,'psk')
    BERtheory[BERtheory==0] = np.nan
    plt.plot(EbN0dB_, np.log10(BERtheory),'-', label=str(M)+'PSK theory')

plt.xlim(min(EbN0dB_), max(EbN0dB_))
plt.ylim(-6, 0)
plt.legend();
plt.xlabel('EbN0 [dB]');
plt.ylabel('log10(BER)');
plt.grid()
# -

# ## Test generalized mutual information (GMI) versus signal-to-noise ratio (SNR)

# ### QAM constellations with Gray mapping

# +
# Run GMI vs SNR Monte Carlo simulation 

qamOrder  = [4, 16, 64, 256, 1024, 4096]  # Modulation order

SNR  = np.arange(-2, 35, 1)
GMI  = np.zeros((len(SNR),len(qamOrder)))

for ii, M in enumerate(qamOrder):
    print('run sim: M = ', M)
    for indSNR in tqdm(range(SNR.size)):

        snrdB = SNR[indSNR]

        # generate random bits
        bitsTx   = np.random.randint(2, size=2**18)    

        # Map bits to constellation symbols
        mod = QAMModem(m=M)
        symbTx = mod.modulate(bitsTx)

        # Normalize symbols energy to 1
        symbTx = symbTx/np.sqrt(mod.Es)

        # AWGN    
        noiseVar = 1/(10**(snrdB/10))

        symbRx = awgn(symbTx, noiseVar)

        # GMI estimation
        GMI[indSNR, ii], _  = monteCarloGMI(symbRx, symbTx, mod)


# +
plt.figure(figsize=(10,6))
for ii, M in enumerate(qamOrder):
    plt.plot(SNR, GMI[:,ii],'-', label=str(M)+'QAM monte carlo',linewidth=2)

# plot theoretical AWGN channel capacity    
C = np.log2(1 + 10**(SNR/10))
plt.plot(SNR, C,'k-', label='AWGN capacity',linewidth=2)


plt.xlim(min(SNR), max(SNR))
plt.legend();
plt.xlabel('SNR [dB]');
plt.ylabel('GMI [bits/symbol]');
plt.grid()
# -
# ### PSK constellations with Gray mapping

# +
# Run GMI vs SNR Monte Carlo simulation 

pskOrder  = [4, 8, 16, 64]  # Modulation order

SNR  = np.arange(-2, 35, 1)
GMI  = np.zeros((len(SNR),len(pskOrder)))

for ii, M in enumerate(pskOrder):
    print('run sim: M = ', M)
    for indSNR in tqdm(range(SNR.size)):

        snrdB = SNR[indSNR]

        # generate random bits
        bitsTx   = np.random.randint(2, size=2**18)    

        # Map bits to constellation symbols
        mod = PSKModem(m=M)
        symbTx = mod.modulate(bitsTx)

        # Normalize symbols energy to 1
        symbTx = symbTx/np.sqrt(mod.Es)

        # AWGN    
        noiseVar = 1/(10**(snrdB/10))

        symbRx = awgn(symbTx, noiseVar)

        # GMI estimation
        GMI[indSNR, ii], _  = monteCarloGMI(symbRx, symbTx, mod)

# +
plt.figure(figsize=(10,6))
for ii, M in enumerate(pskOrder):
    plt.plot(SNR, GMI[:,ii],'-', label=str(M)+'PSK monte carlo',linewidth=2)

# plot theoretical AWGN channel capacity    
C = np.log2(1 + 10**(SNR/10))
plt.plot(SNR, C,'k-', label='AWGN capacity',linewidth=2)


plt.xlim(min(SNR), max(SNR))
plt.legend();
plt.xlabel('SNR [dB]');
plt.ylabel('GMI [bits/symbol]');
plt.grid()
# -

# ## Test mutual information (MI) versus signal-to-noise ratio (SNR)

# ### QAM constellations with Gray mapping

# +
# Run MI vs SNR Monte Carlo simulation 

qamOrder  = [4, 16, 64, 256, 1024]  # Modulation order

SNR  = np.arange(-2, 35, 1)
MI  = np.zeros((len(SNR),len(qamOrder)))

for ii, M in enumerate(qamOrder):
    print('run sim: M = ', M)
         
    mod = QAMModem(m=M)
        
    for indSNR in tqdm(range(SNR.size)):

        snrdB = SNR[indSNR]

        # generate random bits
        bitsTx   = np.random.randint(2, size=2**18)    

        # Map bits to constellation symbols
        symbTx = mod.modulate(bitsTx)

        # Normalize symbols energy to 1
        symbTx = symbTx/np.sqrt(mod.Es)

        # AWGN    
        noiseVar = 1/(10**(snrdB/10))

        symbRx = awgn(symbTx, noiseVar)

        # MI estimation
        MI[indSNR, ii] = monteCarloMI(symbRx, symbTx, mod)

# +
plt.figure(figsize=(10,6))

for ii, M in enumerate(qamOrder):
    plt.plot(SNR, MI[:,ii],'-', label=str(M)+'QAM monte carlo',linewidth=2)

# plot theoretical AWGN channel capacity    
C = np.log2(1 + 10**(SNR/10))
plt.plot(SNR, C,'k-', label='AWGN capacity',linewidth=2)


plt.xlim(min(SNR), max(SNR))
plt.legend();
plt.xlabel('SNR [dB]');
plt.ylabel('MI [bits/symbol]');
plt.grid()
# -
# ## Test mutual information (MI) versus signal-to-noise ratio (SNR) with probabilistically shaped QAM constellation

# +
from numpy.random import choice

def maxwellBolt(λ, const):
    
    p = np.zeros(const.size)
    
    for ind, x in enumerate(const):
        p[ind] = np.exp(-λ*np.abs(x)**2)
        
    p = p/np.sum(p)
    
    return p   

# probSymb = maxwellBolt(1.5, constSymb)

# draw = choice(constSymb, 100000, p=probSymb)

# plt.hist(draw.real, bins=256);


# +
# Run MI vs SNR Monte Carlo simulation 

qamOrder  = [64, 64]  # Modulation order

SNR  = np.arange(-2, 35, 1)
MI  = np.zeros((len(SNR),len(qamOrder)))
Nsymbols = 20000

PS = 1.5
for ii, M in enumerate(qamOrder):
    print('run sim: M = ', M)
          
    mod = QAMModem(m=M)
    constSymb = mod.constellation/np.sqrt(mod.Es)
    
    probSymb = maxwellBolt(PS, constSymb)
    PS = 0
    
    for indSNR in tqdm(range(SNR.size)):

        snrdB = SNR[indSNR]

        # generate random symbols   
        symbTx = choice(constSymb, Nsymbols, p=probSymb)

        # AWGN    
        noiseVar = signal_power(symbTx)/(10**(snrdB/10))

        symbRx = awgn(symbTx, noiseVar)

        # MI estimation
        MI[indSNR, ii] = monteCarloMI(symbRx, symbTx, mod, probSymb)

# +
plt.figure(figsize=(10,6))

for ii, M in enumerate(qamOrder):
    plt.plot(SNR, MI[:,ii],'-', label=str(M)+'QAM monte carlo',linewidth=2)

# plot theoretical AWGN channel capacity    
C = np.log2(1 + 10**(SNR/10))
plt.plot(SNR, C,'k-', label='AWGN capacity',linewidth=2)


plt.xlim(min(SNR), max(SNR))
plt.legend();
plt.xlabel('SNR [dB]');
plt.ylabel('MI [bits/symbol]');
plt.grid()
# -


