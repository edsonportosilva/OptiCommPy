import numpy as np
from commpy.utilities  import signal_power
from commpy.modulation import QAMModem
from numba import njit


@njit
def hardDecision(rxSymb, constSymb, bitMapping):
    
    M = len(constSymb)
    b = int(np.log2(M))
    
    decBits = np.zeros(len(rxSymb)*b)
    
    for i in range(0, len(rxSymb)):       
        indSymb = np.argmin(np.abs(rxSymb[i] - constSymb))        
        decBits[i*b:i*b + b] = bitMapping[indSymb, :]       
                                     
    return decBits

def fastBERcalc(rx, tx, mod):
    """
    BER calculation

    """
    
    # We want all the signal sequences to be disposed in columns:        
    try:
        if rx.shape[1] > rx.shape[0]:
            rx = rx.T       
    except IndexError:
        rx  = rx.reshape(len(rx),1)       
        
    try:        
        if tx.shape[1] > tx.shape[0]:
            tx = tx.T
    except IndexError:        
        tx = tx.reshape(len(tx),1)

    nModes = int(tx.shape[1]) # number of sinal modes
    SNR    = np.zeros(nModes)
    BER    = np.zeros(nModes)

    bitMapping = mod.demodulate(mod.constellation, demod_type = 'hard')
    bitMapping = bitMapping.reshape(-1, int(np.log2(mod.m)))
    
    # pre-processing
    for k in range(0, nModes):
        # symbol normalization
        rx[:,k] = rx[:,k]/np.sqrt(signal_power(rx[:,k]))
        tx[:,k] = tx[:,k]/np.sqrt(signal_power(tx[:,k]))

        # correct (possible) phase ambiguity
        rot = np.mean(tx[:,k]/rx[:,k])
        rx[:,k]  = rot*rx[:,k]
        
        # estimate SNR of the received constellation
        SNR[k] = 10*np.log10(signal_power(tx[:,k])/signal_power(rx[:,k]-tx[:,k]))
        
    for k in range(0, nModes):     
        # hard decision demodulation of the received symbols   
        brx = hardDecision(np.sqrt(mod.Es)*rx[:,k], mod.constellation, bitMapping)
        btx = hardDecision(np.sqrt(mod.Es)*tx[:,k], mod.constellation, bitMapping)
        
        err = np.logical_xor(brx, btx)
        BER[k] = np.mean(err)
        
    return BER, SNR

@njit
def calcLLR(rxSymb, M, σ2, constSymb, bitMapping):
    
    b = int(np.log2(M))
    
    LLRs = np.zeros(len(rxSymb)*b)
    
    for i in range(0, len(rxSymb)):       
        prob = np.exp((-np.abs(rxSymb[i] - constSymb)**2)/σ2)               
        
        for indBit in range(0, b):
            p0  = np.sum(prob[bitMapping[:,indBit]==0])
            p1  = np.sum(prob[bitMapping[:,indBit]==1])
            
            LLRs[i*b + indBit] = np.log(p0)-np.log(p1)          
                                 
    return LLRs

def monteCarloGMI(rx, tx, mod):
    """
    GMI calculation
    """
    
    # We want all the signal sequences to be disposed in columns:        
    try:
        if rx.shape[1] > rx.shape[0]:
            rx = rx.T       
    except IndexError:
        rx  = rx.reshape(len(rx),1)       
        
    try:        
        if tx.shape[1] > tx.shape[0]:
            tx = tx.T
    except IndexError:        
        tx = tx.reshape(len(tx),1)

    nModes = int(tx.shape[1]) # number of sinal modes
    GMI    = np.zeros(nModes)
    
    noiseVar = np.var(rx-tx, axis=0)
    
    bitMapping = mod.demodulate(mod.constellation, demod_type = 'hard')
    bitMapping = bitMapping.reshape(-1, int(np.log2(mod.m)))
        
    # symbol normalization
    for k in range(0, nModes):       
        rx[:,k] = rx[:,k]/np.sqrt(signal_power(rx[:,k]))
        tx[:,k] = tx[:,k]/np.sqrt(signal_power(tx[:,k]))
        
    for k in range(0, nModes):        
        # set the noise variance 
        σ2 = noiseVar[k]
        
        # hard decision demodulation of the transmitted symbols
        btx = mod.demodulate(np.sqrt(mod.Es)*tx[:,k], demod_type = 'hard')
                        
        # soft demodulation of the received symbols                       
        LLRs = calcLLR(rx[:,k], mod.m, σ2, mod.constellation/np.sqrt(mod.Es), bitMapping) 
                
        LLRs[LLRs == np.inf] = 500
        LLRs[LLRs == -np.inf] = -500
    
        # Compute bitwise MIs and their sum
        b = int(np.log2(mod.m))
        
        MIperBitPosition = np.zeros(b)
               
        for n in range(0, b):
            MIperBitPosition[n] = 1 - np.mean(np.log2(1 + np.exp( (2*btx[n::b]-1)*LLRs[n::b]) ) )
                        
        GMI[k] = np.sum(MIperBitPosition)
                
    return GMI, MIperBitPosition
