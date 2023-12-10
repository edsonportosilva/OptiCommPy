# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# <a href="https://colab.research.google.com/github/edsonportosilva/OptiCommPy/blob/main/jupyter/test_WDM_transmission.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Simulation of coherent WDM transmission

if 'google.colab' in str(get_ipython()):    
    # ! git clone -b main https://github.com/edsonportosilva/OptiCommPy
    from os import chdir as cd
    cd('/content/OptiCommPy/')
    # ! pip install . 

# +
import matplotlib.pyplot as plt
import numpy as np

from optic.dsp.core import pulseShape, firFilter, decimate, symbolSync, pnorm, signal_power
from optic.models.devices import pdmCoherentReceiver, basicLaserModel

try:
    from optic.models.modelsGPU import manakovSSF
except:
    from optic.models.channels import manakovSSF

from optic.models.tx import simpleWDMTx
from optic.utils import parameters
from optic.dsp.equalization import edc, mimoAdaptEqualizer
from optic.dsp.carrierRecovery import cpr
from optic.comm.metrics import fastBERcalc, monteCarloGMI, monteCarloMI, calcEVM
from optic.plot import pconst

import scipy.constants as const

from optic.models.amplification import edfaSM,OSA

import os.path as path
# -

import logging as logg
logg.getLogger().setLevel(logg.INFO)
logg.basicConfig(format='%(message)s')

from IPython.core.display import HTML
from IPython.core.pylabtools import figsize

# +
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")

figsize(10, 3)
# -

# %load_ext autoreload
# %autoreload 2
# %load_ext line_profiler

# + [markdown] magic_args="[markdown]"
#
# ## Parameters
# -

# Transmitter parameters:
paramTx = parameters()
paramTx.M   = 16           # order of the modulation format
paramTx.Rs  = 32e9         # symbol rate [baud]
paramTx.SpS = 16           # samples per symbol
paramTx.pulse = 'rrc'      # pulse shaping filter
paramTx.Ntaps = 2*4096     # number of pulse shaping filter coefficients
paramTx.alphaRRC = 0.01    # RRC rolloff
paramTx.Pch_dBm = -2        # power per WDM channel [dBm]
paramTx.Nch     = 11       # number of WDM channels
paramTx.Fc      = 193.1e12 # central optical frequency of the WDM spectrum
paramTx.lw      = 100e3    # laser linewidth in Hz
paramTx.freqSpac = 37.5e9  # WDM grid spacing
paramTx.Nmodes = 2         # number of signal modes [2 for polarization multiplexed signals]
paramTx.Nbits = int(np.log2(paramTx.M)*1e5) # total number of bits per polarization

# +
# Optical channel parameters:
paramCh = parameters()
paramCh.Ltotal = 100      # total link distance [km]
paramCh.Lspan  = 100      # span length [km]
paramCh.alpha  = 0.2     # fiber loss parameter [dB/km]
paramCh.D = 16           # fiber dispersion parameter [ps/nm/km]
paramCh.gamma = 1.3      # fiber nonlinear parameter [1/(W.km)]
paramCh.Fc = paramTx.Fc  # central optical frequency of the WDM spectrum
paramCh.hz = 0.5         # step-size of the split-step Fourier method [km]
paramCh.maxIter = 5      # maximum number of convergence iterations per step
paramCh.tol = 1e-5       # error tolerance per step
paramCh.prgsBar = True   # show progress bar?
paramCh.amp = None
paramCh.Fs = paramTx.Rs*paramTx.SpS # sampling rate

Fs = paramCh.Fs
# -

# EDFA parameters
param_edfa = parameters()
# gain control
param_edfa.type     = "AGC"
param_edfa.value    = 20
# pid - gain control
param_edfa.kp = 1e-2
param_edfa.ki = 1e-2
param_edfa.kd = 5e-2
# pump information
param_edfa.forPump  = {'pump_signal': np.array([100e-3]), 'pump_lambda': np.array([980e-9])}
param_edfa.bckPump  = {'pump_signal': np.array([000e-3]), 'pump_lambda': np.array([980e-9])}
# giles parameters
param_edfa.file     = 'giles_MP980.dat'
param_edfa.fileunit = 'nm'
param_edfa.gmtc     = 'Bessel'
param_edfa.tol      = 3/100
param_edfa.tolCtrl  = 0.5
param_edfa.noiseBand= 12.5e9

if 'google.colab' in str(get_ipython()):  
  param_edfa.file = path.join(path.abspath(path.join("../")), 'OptiCommPy', 'optic', 'models', 'ampParams', param_edfa.file)
else:
  param_edfa.file = path.join(path.abspath(path.join("../")), 'optic', 'models', 'ampParams', param_edfa.file)

# ## Simulation

# ### Singal generation, propagation, amplification and reception

# **Polarization multiplexed WDM signal generation**

# generate WDM signal
sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)

print('Average power of the modulated optical signal [mW]: %.3f mW'%(signal_power(sigWDM_Tx)/1e-3))
print('Average power of the modulated optical signal [dBm]: %.3f dBm'%(10*np.log10(signal_power(sigWDM_Tx)/1e-3)))

ax = OSA(sigWDM_Tx, Fs, paramTx.Fc)
ax.set_xlim([1550,1555])
ax.set_ylim([-80,-40])

# **Nonlinear fiber propagation with the split-step Fourier method**

# nonlinear signal propagation
sigWDM = manakovSSF(sigWDM_Tx, paramCh)

# +

ax = OSA(sigWDM, Fs, paramTx.Fc)
ax.set_xlim([1550,1555])
ax.set_ylim([-80,-40])

# +

prog_pw = signal_power(sigWDM)/1e-3 #np.sum(1000*np.mean(sigWDM * np.conj(sigWDM), axis = 0).real)
print('Average power of the modulated optical signal [mW]: %.3f mW'%(prog_pw))
print('Average power of the modulated optical signal [dBm]: %.3f dBm'%(10*np.log10(prog_pw)))
# -

# **Optical amplification using an EDFA**

sigWDM_Amp, PumpF, PumpB, noisef  = edfaSM(sigWDM, Fs, paramTx.Fc, param_edfa)

rx_pw = 1e3*signal_power(sigWDM_Amp)
print('Forward pump  - [mW] : %.3f' %(1e3*PumpF[0]))
print('Backward pump - [mW] : %.3f' %(1e3*PumpB[1]))
print('Average power - RX amp [mW] : %.3f' %(rx_pw))
print('Average power - RX amp [dBm] : %.3f' %(10*np.log10(rx_pw)))
print('Gain [dB]: %.3f' %(10*np.log10(rx_pw/prog_pw)))

# +

ax = OSA(sigWDM_Amp, Fs, paramTx.Fc)
ax.set_xlim([1550,1555])
ax.set_ylim([-80,-40])
# -

# **Optical WDM spectrum before and after transmission**

# plot psd
plt.figure(figsize=(10, 3))
plt.xlim(paramCh.Fc-Fs/2,paramCh.Fc+Fs/2);
plt.psd(sigWDM_Tx[:,0], Fs=Fs, Fc=paramCh.Fc, NFFT = 4*1024, sides='twosided', label = 'WDM spectrum - Tx')
plt.psd(sigWDM[:,0], Fs=Fs, Fc=paramCh.Fc, NFFT = 4*1024, sides='twosided', label = 'WDM spectrum - Rx')
plt.psd(sigWDM_Amp[:,0], Fs=Fs, Fc=paramCh.Fc, NFFT = 4*1024, sides='twosided', label = 'WDM spectrum - Amp - Rx')
plt.legend(loc='lower left')
plt.title('optical WDM spectrum');

# **WDM channels coherent detection and demodulation**

# +
# parameters
chIndex  = int(np.floor(paramTx.Nch/2))      # index of the channel to be demodulated

freqGrid = paramTx.freqGrid
π  = np.pi
t  = np.arange(0, len(sigWDM))*1/Fs

print('Demodulating channel #%d , fc: %.4f THz, λ: %.4f nm\n'\
      %(chIndex, (paramCh.Fc + freqGrid[chIndex])/1e12, const.c/(paramCh.Fc + freqGrid[chIndex])/1e-9))

symbTx = symbTx_[:,:,chIndex]

# local oscillator (LO) parameters:
FO      = 150e6                 # frequency offset
Δf_lo   = freqGrid[chIndex]+FO  # downshift of the channel to be demodulated

# generate CW laser LO field
paramLO = parameters()
paramLO.P = 10              # power in dBm
paramLO.lw = 100e3          # laser linewidth
paramLO.RIN_var = 0
paramLO.Ns = len(sigWDM)
paramLO.Fs = Fs

sigLO = basicLaserModel(paramLO)
sigLO = sigLO*np.exp(1j*2*π*Δf_lo*t) # add frequency offset  

print('Local oscillator P: %.2f dBm, lw: %.2f kHz, FO: %.2f MHz\n'\
      %(paramLO.P, paramLO.lw/1e3, FO/1e6))
# -

# **polarization multiplexed coherent optical receiver**

# +
# photodiodes parameters
paramPD = parameters()
paramPD.B = paramTx.Rs
paramPD.Fs = Fs    
paramPD.ideal = True

θsig = π/3 # polarization rotation angle
sigRx = pdmCoherentReceiver(sigWDM_Amp, sigLO, θsig, paramPD)

# plot received constellations
pconst(sigRx[0::paramTx.SpS,:], R=3);
# -

# **Matched filtering and CD compensation**

# +
# Matched filtering
if paramTx.pulse == 'nrz':
    pulse = pulseShape('nrz', paramTx.SpS)
elif paramTx.pulse == 'rrc':
    pulse = pulseShape('rrc', paramTx.SpS, N=paramTx.Ntaps, alpha=paramTx.alphaRRC, Ts=1/paramTx.Rs)

pulse = pnorm(pulse)
sigRx = firFilter(pulse, sigRx)

# plot constellations after matched filtering
pconst(sigRx[0::paramTx.SpS,:], R=3);

# +
# CD compensation
paramEDC = parameters()
paramEDC.L = paramCh.Ltotal
paramEDC.D = paramCh.D
paramEDC.Fc = paramCh.Fc-Δf_lo
paramEDC.Fs = Fs

sigRx = edc(sigRx, paramEDC)

# plot constellations after CD compensation
pconst(sigRx[0::paramTx.SpS,:], R=3);
# -

# **Downsampling to 2 samples/symbol and re-synchronization with transmitted sequences**

# +
# decimation
paramDec = parameters()
paramDec.SpS_in  = paramTx.SpS
paramDec.SpS_out = 2
sigRx = decimate(sigRx, paramDec)

symbRx = symbolSync(sigRx, symbTx, 2)
# -

# **Power normalization**

x = pnorm(sigRx)
d = pnorm(symbRx)

# **Adaptive equalization**

# +
# adaptive equalization parameters
paramEq = parameters()
paramEq.nTaps = 15
paramEq.SpS = paramDec.SpS_out
paramEq.numIter = 5
paramEq.storeCoeff = False
paramEq.M = paramTx.M
paramEq.L = [int(0.2*d.shape[0]), int(0.8*d.shape[0])]
paramEq.prgsBar = False

if paramTx.M == 4:
    paramEq.alg = ['nlms','cma'] # QPSK
    paramEq.mu = [5e-3, 1e-3] 
else:
    paramEq.alg = ['da-rde','rde'] # M-QAM
    paramEq.mu = [5e-3, 2e-4] 

y_EQ = mimoAdaptEqualizer(x, paramEq, d)

#plot constellations after adaptive equalization
discard = 5000
pconst(y_EQ[discard:-discard,:]);
# -

# **Carrier phase recovery**

# +
paramCPR = parameters()
paramCPR.alg = 'bps'
paramCPR.M   = paramTx.M
paramCPR.N   = 75
paramCPR.B   = 64

y_CPR = cpr(y_EQ, paramCPR)

discard = 500

#plot constellations after CPR
pconst([y_CPR[discard:-discard,:],d[discard:-discard,:]], pType='fast')
pconst(y_CPR[discard:-discard,:]);
# -

# ### Evaluate transmission metrics

# +
ind = np.arange(discard, d.shape[0]-discard)
BER, SER, SNR = fastBERcalc(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
GMI, NGMI = monteCarloGMI(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
MI       = monteCarloMI(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
EVM      = calcEVM(y_CPR[ind,:], paramTx.M, 'qam', d[ind,:])

print('      pol.X      pol.Y      ')
print(' SER: %.2e,  %.2e'%(SER[0], SER[1]))
print(' BER: %.2e,  %.2e'%(BER[0], BER[1]))
print(' SNR: %.2f dB,  %.2f dB'%(SNR[0], SNR[1]))
print(' EVM: %.2f %%,    %.2f %%'%(EVM[0]*100, EVM[1]*100))
print('  MI: %.2f bits, %.2f bits'%(MI[0], MI[1]))
print(' GMI: %.2f bits, %.2f bits'%(GMI[0], GMI[1]))
print('NGMI: %.2f,      %.2f'%(NGMI[0], NGMI[1]))
