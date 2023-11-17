# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# <a href="https://colab.research.google.com/github/edsonportosilva/OptiCommPy/blob/main/examples/test_modulation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Test basic DSP functionalities

if 'google.colab' in str(get_ipython()):    
    # ! git clone -b main https://github.com/edsonportosilva/OptiCommPy
    from os import chdir as cd
    cd('/content/OptiCommPy/')
    # ! pip install . 

from optic.dsp.core import pnorm, signal_power, decimate, resample, lowPassFIR, firFilter, clockSamplingInterp, quantizer, upsample, pulseShape
from optic.utils import parameters
from optic.plot import eyediagram, plotPSD, pconst
from optic.comm.modulation import modulateGray
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

# %load_ext autoreload
# %autoreload 2

# ## Test rational resample

# +
Fs = 400
fc = 100

t = np.arange(0,4096)*(1/Fs)
π = np.pi

sig = np.sin(2*π*fc*t)

plt.plot(t, sig,'-o',markersize=4);
plt.xlim(0, 10*1/fc)

paramDec = parameters()
paramDec.SpS_in = 4
paramDec.SpS_out = 16
paramDec.Rs = fc

resFactor = paramDec.SpS_out/paramDec.SpS_in

t_dec = np.arange(0, int(resFactor*4096))*(1/(resFactor*Fs))
sig_dec = resample(sig, paramDec)

plt.plot(t_dec, sig_dec,'-o',markersize=4);
plt.xlim(0, 10*1/fc)
# -
# ## Test sampling clock converter

# +
Fs = 3200
fc = 100

t = np.arange(0, 300000)*(1/Fs)
π = np.pi

# generate sinusoidal signal
sig = np.sin(2*π*fc*t)

plt.plot(t, sig,'-o',markersize=4);
plt.xlim(min(t), max(t))

# intermpolate signal to a given clock sampling frequency and jitter
Fs_in = Fs
Fs_out = 1.001*Fs
AAF = False
jitter_rms = 1e-9

t_dec = clockSamplingInterp(t.reshape(-1,1), Fs_in, Fs_out, jitter_rms)
sig_dec = clockSamplingInterp(sig.reshape(-1,1), Fs_in, Fs_out, jitter_rms)
plt.plot(t_dec, sig_dec,'-o',markersize=4);
plt.xlim(0, 10*1/fc)

eyediagram(sig_dec.reshape(-1,), sig_dec.size, int(Fs//fc), n=3, ptype='fast', plotlabel=None)
# -
# ## Test signal quantizer

# +
Fs = 3200
fc = 100

t = np.arange(0, 300000)*(1/Fs)
π = np.pi

# generate sinusoidal signal
sig = np.sin(2*π*fc*t) #+ np.sin(6*π*fc*t)/3 + np.sin(10*π*fc*t)/5

plt.plot(t, sig,'-o',markersize=4);
plt.xlim(min(t), max(t))

# quantizer
nBits = 2
sig_q = quantizer(sig.reshape(-1,1), nBits)

plt.plot(t, sig_q,'--k',markersize=4);
plt.xlim(0, 10*1/fc)

eyediagram(sig_dec.reshape(-1,), sig_dec.size, int(Fs//fc), n=3, ptype='fast', plotlabel=None)


# -

def adc(Ei, param):
    """
    Analog-to-digital converter (ADC) model.

    Parameters
    ----------
    Ei : ndarray
        Input signal.
    param : core.parameter
        Resampling parameters:
            param.Fs_in  : sampling frequency of the input signal.
            param.Fs_out : sampling frequency of the output signal.
            param.jitter_rms : root mean square (RMS) value of the jitter in seconds.
            param.nBits : number of bits used for quantization.
            param.Vmax : maximum value for the ADC's full-scale range.
            param.Vmin : minimum value for the ADC's full-scale range.
            param.AAF : flag indicating whether to use anti-aliasing filters (True/False).
            param.N : number of taps of the anti-aliasing filters.

    Returns
    -------
    Eo : ndarray
        Resampled and quantized signal.

    """
    # Check and set default values for input parameters
    param.Fs_in = getattr(param, "Fs_in", 1)
    param.Fs_out = getattr(param, "Fs_out", 1)
    param.jitter_rms = getattr(param, "jitter_rms", 0)
    param.nBits = getattr(param, "nBits", 16)
    param.Vmax = getattr(param, "Vmax", 1)
    param.Vmin = getattr(param, "Vmin", -1)
    param.AAF = getattr(param, "AAF", True)
    param.N   = getattr(param, "N", 202)

    # Extract individual parameters for ease of use
    Fs_in = param.Fs_in
    Fs_out = param.Fs_out
    jitter_rms = param.jitter_rms
    nBits = param.nBits
    Vmax = param.Vmax
    Vmin = param.Vmin
    AAF = param.AAF
    N = param.N

    # Reshape the input signal if needed to handle single-dimensional inputs
    try:
        Ei.shape[1]
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)

    # Get the number of modes (columns) in the input signal
    nModes = Ei.shape[1]

    # Apply anti-aliasing filters if AAF is enabled
    if AAF:
        # Anti-aliasing filters:
        Ntaps = min(Ei.shape[0], N)
        hi = lowPassFIR(param.Fs_out/2, param.Fs_in, Ntaps, typeF="rect")
        ho = lowPassFIR(param.Fs_in/2, param.Fs_out, Ntaps, typeF="rect")

        Ei = firFilter(hi, Ei)

    # Signal interpolation to the ADC's sampling frequency
    Eo = clockSamplingInterp(Ei, Fs_in, Fs_out, jitter_rms)

    # Uniform quantization of the signal according to the number of bits of the ADC
    Eo = quantizer(Eo, nBits, Vmax, Vmin)

    # Apply anti-aliasing filters to the output if AAF is enabled
    if AAF:
        Eo = firFilter(ho, Eo)

    return Eo


# +
Fs = 3200
fc = 100

t = np.arange(0, 10000)*(1/Fs)
π = np.pi

# generate sinusoidal signal
sig = np.array([np.sin(2*π*fc*t), np.cos(2*π*fc*t), np.sin(2*π*fc*t)+np.cos(2*π*fc*t)]).T

# ADC input parameters
param = parameters()
param.Fs_in = Fs
param.Fs_out = Fs
param.jitter_rms = 5e-5
param.nBits =  16
param.Vmax = 1
param.Vmin = -1
param.AAF = True
param.N = 512

sig_adc = adc(sig, param)

plt.plot(sig_adc)
plt.xlim(0,100);

eyediagram(sig_adc[:,0], sig_adc.shape[0], int(param.Fs_out//fc), n=3, ptype='fast', plotlabel=None)

# +
import numpy as np
from scipy import interpolate

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def gardner_ted(signal):      
    return signal[1]*(signal[2]-signal[0]) 

def interpolator(sigIn, tnco):
    # Cubic interpolator with Farrow architecture:
    sigOut =  sigIn[0]*(-1/6 * tnco**3 + 1/6 * tnco + 0) +\
              sigIn[1]*(1/2  * tnco**3 + 1/2 * tnco**2 -   1 * tnco) +\
              sigIn[2]*(-1/2 * tnco**3 - 1 * tnco**2 + 1/2 * tnco + 1) +\
              sigIn[3]*(1/6  * tnco**3 + 1/2 * tnco**2 + 1/3 * tnco)    

    return sigOut

def clockRecovery(sigIn, kp = 1e-3, ki=1e-4):
    
    # Initializing variables:    
    intPart = 0
    t_nco = 0
    
    L = len(sigIn)    
    outInterp = sigIn.copy()
    
    ted = 0
    loopFilterOut = 0
    ted_values = []
    
    n = 2
    m = 2
 
    while n < L-1 and m < L-2:               
        outInterp[n] = interpolator(sigIn[m-2:m+2], t_nco)
        
        if n%2==0:
            ted = gardner_ted(outInterp[n-2:n+1])
            
            # Loop PI Filter:
            intPart = ki* ted + intPart
            propPart = kp * ted
            loopFilterOut = propPart + intPart
            
            t_nco -= loopFilterOut
        
        n += 1
        m += 1
        
        # NCO
        if t_nco > 0:
            t_nco -= 1
            m -= 1
            
        elif t_nco < -1:
            t_nco += 1
            m += 1
            
        ted_values.append(t_nco)    
                            
    return outInterp, ted_values


# +
# simulation parameters
SpS = 16           # samples per symbol
M = 2              # order of the modulation format
Rs = 10e9          # Symbol rate (for OOK case Rs = Rb)
Fs = SpS*Rs        # Sampling frequency in samples/second
Ts = 1/Fs          # Sampling period

# generate pseudo-random bit sequence
bitsTx = np.random.randint(2, size=int(np.log2(M)*64e3))

# generate ook modulated symbol sequence
symbTx = modulateGray(bitsTx, M, 'pam')    
symbTx = pnorm(symbTx) # power normalization

# upsampling
symbolsUp = upsample(symbTx, SpS)

# typical NRZ pulse
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse))

# pulse shaping
sigTx = firFilter(pulse, symbolsUp)
downSample = 7.999
sigRx = clockSamplingInterp(sigTx.reshape(-1,1), Fs, Fs/downSample, 0)
sigRxRef = clockSamplingInterp(sigTx.reshape(-1,1), Fs, Fs/8, 0)


outCLK, ted_values = clockRecovery(sigRx.reshape(-1,), kp=1e-3, ki=1e-6)#print("TED Values:", ted_values)
0

plt.plot(sigRx,'k.', label=f'SpS = {SpS/downSample}')
plt.plot(outCLK,'b.', label= 'Out clock recovery')
plt.plot(sigRxRef,'r.', label='SpS = 2')
plt.xlabel('sample')
plt.grid()
plt.xlim([0, len(sigRx)])
plt.legend()

plt.figure()
plt.plot( moving_average(ted_values, 10), label = 'timing')
plt.xlabel('sample')
plt.grid()
plt.xlim([0, len(sigRx)])
plt.legend()

# plotPSD( ted_values.reshape(-1,), Fs=Fs/8, Fc=0, NFFT=4096)
# plt.xlim([0, Fs/200])
# -

pconst(outCLK[200:-2:2],pType='fast')

# +
import numpy as np

def interpolator(sigIn, t):
    # In[mn-2]: sigIn[0]
    # In[mn-1]: sigIn[1]
    # In[mn]: sigIn[2]
    # In[mn+1]: sigIn[3]
    
        # Cubic interpolator with Farrow architecture:
        sigOut = (
            sigIn[0]*(-1/6 * mun**3 + 0 * mun**2 + 1/6 * mun + 0) +
            sigIn[1]*(1/2 * mun**3 + 1/2 * mun**2 - 1 * mun + 0) +
            sigIn[2]*(-1/2 * mun**3 - 1 * mun**2 + 1/2 * mun + 1) +
            sigIn[3]*(1/6 * mun**3 + 1/2 * mun**2 + 1/3 * mun + 0)
        )

    return sigOut

# Example usage:
parallelization_value = 4
timing_value = 0.01
buffer_pointer_value = 4

input_data = np.array([1, 2, 3, 4, 5, 6])
buffer_data = np.zeros(11)
result, updated_buffer = interpolator(input_data, buffer_data, buffer_pointer_value, parallelization_value, timing_value)

print("Result:", result)
print("Updated Buffer:", updated_buffer)



# +
import numpy as np

def clock_recovery(In, PSType, NSymb, ParamCR):
    ki = ParamCR['ki']
    kp = ParamCR['kp']

    Etamn = 0.5
    Wk = 1
    LF_I = Wk
    mun = 0
    n = 3
    mn = n
    LIn = len(In)

    Out = np.zeros_like(In)
    Out[:3] = In[:3]

    if ParamCR['DPLLVarOut']:
        WkVec = []
        EtamnVec = []
        munVec = []
        mnVec = []
        ekVec = []

    while mn <= LIn:
        if mn == LIn:
            In[mn] = 0

        # Cubic interpolator with Farrow architecture:
        Out[n] = (
            In[mn - 2] * (-1 / 6 * mun**3 + 0 * mun**2 + 1 / 6 * mun + 0) +
            In[mn - 1] * (1 / 2 * mun**3 + 1 / 2 * mun**2 - 1 * mun + 0) +
            In[mn]     * (-1 / 2 * mun**3 - 1 * mun**2 + 1 / 2 * mun + 1) +
            In[mn + 1] * (1 / 6 * mun**3 + 1 / 2 * mun**2 + 1 / 3 * mun + 0)
        )

        # Generating the Ts-spaced timing error indication signal 'e_n':
        if n % 2 == 1:
            if PSType == 'Nyquist':
                # TED - Nyquist pulses:
                ek = np.abs(Out[n - 1])**2 * (np.abs(Out[n - 2])**2 - np.abs(Out[n])**2)
            elif PSType == 'NRZ':
                # TED - NRZ pulses:
                ek = np.real(np.conj(Out[n - 1]) * (Out[n] - Out[n - 2]))

            # Loop Filter:
            LF_I = ki * ek + LF_I
            LF_P = kp * ek
            Wk = LF_P + LF_I

            # Parameters of the DPLL:
            if ParamCR['DPLLVarOut']:
                WkVec.append(Wk)
                EtamnVec.append(Etamn)
                munVec.append(mun)
                mnVec.append(mn)
                ekVec.append(ek)

        # NCO - Base point 'mk' and fractional interval 'mu_n':
        if -1 < (Etamn - Wk) and (Etamn - Wk) < 0:
            mn = mn + 1
        elif (Etamn - Wk) >= 0:
            mn = mn + 2

        Etamn = np.mod(Etamn - Wk, 1)
        mun = Etamn / Wk

        # Updating the temporal index 'n':
        n = n + 1

    # Limiting the length of the output to NSymb*2:
    if NSymb * 2 < len(Out):
        Out = Out[:NSymb * 2]
    else:
        Out = Out

    # Parameters of the DPLL:
    if ParamCR['DPLLVarOut']:
        return Out, np.array(WkVec), np.array(EtamnVec), np.array(munVec), np.array(mnVec), np.array(ekVec)
    else:
        return Out

# -


