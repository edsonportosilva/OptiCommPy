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
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from commpy.utilities  import signal_power, upsample
from commpy.modulation import QAMModem
from utils.dsp import firFilter, pulseShape, lowPassFIR, edc, fourthPowerFOE, dbp, cpr
from utils.models import mzm, linFiberCh, iqm, ssfm, edfa, phaseNoise, coherentReceiver, manakov_ssf
from utils.tx import simpleWDMTx
from utils.core import parameters

from scipy import signal
import scipy.constants as const

# +
from IPython.core.display import HTML
from IPython.core.pylabtools import figsize
from IPython.display import display, Math

HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")
# -

# %matplotlib inline
#figsize(7, 2.5)
figsize(10, 3)

# # Simulation of coherent WDM systems

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Coherent-WDM-system" data-toc-modified-id="Coherent-WDM-system-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Coherent WDM system</a></span><ul class="toc-item"><li><span><a href="#Transmitter" data-toc-modified-id="Transmitter-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Transmitter</a></span></li></ul></li></ul></div>
# -

# ## Coherent WDM system
#
# ### Transmitter

help(simpleWDMTx)

help(ssfm)

# **WDM signal generation**

# +
# Parâmetros do transmissor:
param = parameters()
param.M   = 16           # ordem do formato de modulação
param.Rs  = 32e9         # taxa de sinalização [baud]
param.SpS = 16           # número de amostras por símbolo
param.Nbits = 60000      # número de bits
param.pulse = 'rrc'      # formato de pulso
param.Ntaps = 4096       # número de coeficientes do filtro RRC
param.alphaRRC = 0.01    # rolloff do filtro RRC
param.Pch_dBm = 0        # potência média por canal WDM [dBm]
param.Nch     = 5        # número de canais WDM
param.Fc      = 193.1e12 # frequência central do espectro WDM
param.freqSpac = 40e9    # espaçamento em frequência da grade de canais WDM
param.Nmodes = 1         # número de modos de polarização

sigWDM_Tx, symbTx_, param = simpleWDMTx(param)

freqGrid = param.freqGrid
# -
# **Nonlinear fiber propagation with the split-step Fourier method**

# +
linearChannel = False

# optical channel parameters
paramCh = parameters()
paramCh.Ltotal = 800   # km
paramCh.Lspan  = 80    # km
paramCh.alpha = 0.2    # dB/km
paramCh.D = 16         # ps/nm/km
paramCh.Fc = 193.1e12  # Hz
paramCh.hz = 0.5       # km
paramCh.gamma = 1.3    # 1/(W.km)

if linearChannel:
    paramCh.hz = paramCh.Lspan  # km
    paramCh.gamma = 0   # 1/(W.km)

Fs = param.Rs*param.SpS
sigWDM, paramCh = ssfm(sigWDM_Tx, Fs, paramCh) 
# -

# **Optical WDM spectrum before and after transmission**

# plot psd
plt.figure()
plt.xlim(paramCh.Fc-Fs/2,paramCh.Fc+Fs/2);
plt.psd(sigWDM_Tx[:,0], Fs=param.SpS*param.Rs, Fc=paramCh.Fc, NFFT = 4*1024, sides='twosided', label = 'WDM spectrum - Tx')
plt.psd(sigWDM, Fs=Fs, Fc=paramCh.Fc, NFFT = 4*1024, sides='twosided', label = 'WDM spectrum - Rx')
plt.legend(loc='lower left')
plt.title('optical WDM spectrum');


# **WDM channels coherent detection and demodulation**

# +
### Receiver

# parameters
chIndex = 2    # index of the channel to be demodulated
plotPSD = True

Fa = param.SpS*param.Rs
Fc = paramCh.Fc
Ta = 1/Fa
mod = QAMModem(m=param.M)

print('Demodulating channel #%d , fc: %.4f THz, λ: %.4f nm\n'\
      %(chIndex, (Fc + freqGrid[chIndex])/1e12, const.c/(Fc + freqGrid[chIndex])/1e-9))

sigWDM = sigWDM.reshape(len(sigWDM),)
symbTx = symbTx_[:,:,chIndex].reshape(len(symbTx_),)

# local oscillator (LO) parameters:
FO      = 64e6                 # frequency offset
Δf_lo   = freqGrid[chIndex]+FO  # downshift of the channel to be demodulated
lw      = 100e3                 # linewidth
Plo_dBm = 10                    # power in dBm
Plo     = 10**(Plo_dBm/10)*1e-3 # power in W
ϕ_lo    = 0                     # initial phase in rad    

print('Local oscillator P: %.2f dBm, lw: %.2f kHz, FO: %.2f MHz\n'\
      %(Plo_dBm, lw/1e3, FO/1e6))

# generate LO field
π       = np.pi
t       = np.arange(0, len(sigWDM))*Ta
ϕ_pn_lo = phaseNoise(lw, len(sigWDM), Ta)
sigLO   = np.sqrt(Plo)*np.exp(1j*(2*π*Δf_lo*t + ϕ_lo + ϕ_pn_lo))

# single-polarization coherent optical receiver
sigRx = coherentReceiver(sigWDM, sigLO)

# Rx filtering

# Matched filter
if param.pulse == 'nrz':
    pulse = pulseShape('nrz', param.SpS)
elif param.pulse == 'rrc':
    pulse = pulseShape('rrc', param.SpS, N=param.Ntaps, alpha=param.alphaRRC, Ts=1/param.Rs)

pulse = pulse/np.max(np.abs(pulse))            
sigRx = firFilter(pulse, sigRx)

# plot psd
if plotPSD:
    plt.figure();
   # plt.ylim(-250,-50);
    plt.psd(sigRx, Fs=Fa, NFFT = 16*1024, sides='twosided', label = 'Spectrum of the received signal')
    plt.legend(loc='upper left');
    plt.xlim(-Fa/2,Fa/2);

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(15,4.5))
fig.suptitle('Sequence of DSP blocks')

ax1.plot(sigRx.real, sigRx.imag,'.', markersize=4)
ax1.axis('square')
ax1.title.set_text('Output of coherent front-end')
ax1.grid()

# digital backpropagation
hzDBP = 5
Pin   = 10**(param.Pch_dBm/10)*1e-3
sigRx = sigRx/np.sqrt(signal_power(sigRx))
sigRx = dbp(np.sqrt(Pin)*sigRx, Fa, paramCh.Ltotal, paramCh.Lspan, hzDBP, paramCh.alpha, paramCh.gamma, paramCh.D, paramCh.Fc)
sigRx = sigRx.reshape(len(sigRx),)
sigRx = firFilter(pulse, sigRx)
    
# CD compensation
# sigRx = edc(sigRx, paramCh.Ltotal, paramCh.D, Fc-Δf_lo, Fa)

# simple timing recovery
varVector = np.var((sigRx.T).reshape(-1,param.SpS), axis=0) # finds best sampling instant
sampDelay = np.where(varVector == np.amax(varVector))[0][0]

# downsampling
sigRx = sigRx[sampDelay::param.SpS]

discard = 1000
ind = np.arange(discard, sigRx.size-discard)

# symbol normalization
sigRx = sigRx/np.sqrt(signal_power(sigRx[ind]))

# plot constellation after CD compensation
ax2.plot(sigRx.real, sigRx.imag,'.', markersize=4)
ax2.axis('square')
ax2.title.set_text('After CD comp.')
ax2.grid()

# calculate time delay due to walkoff
symbDelay = np.argmax(signal.correlate(np.abs(symbTx), np.abs(sigRx)))-sigRx.size+1 

# compensate walkoff time delay
sigRx = np.roll(sigRx, symbDelay)

# symbol normalization
sigRx = sigRx/np.sqrt(signal_power(sigRx[ind]))

# estimate and compensate LO frequency offset
fo = fourthPowerFOE(sigRx, 1/param.Rs)
print('Estimated FO : %3.4f MHz'%(fo/1e6))

sigRx = sigRx*np.exp(-1j*2*π*fo*np.arange(0,len(sigRx))/param.Rs)

# plot constellation after LO frequency offset compensation
ax3.plot(sigRx[ind].real, sigRx[ind].imag,'.', markersize=4)
ax3.axis('square')
ax3.title.set_text('After CFR (4th-power FOE)')
ax3.grid()

# compensate phase noise (carrier phase recovery - cpr)
windowSize = 40
c  = mod.constellation/np.sqrt(mod.Es)
sigRx, ϕ, θ = cpr(sigRx, windowSize, c, symbTx)

# plot phases estimated by cpr
phaseOffSet = np.mean(np.roll(ϕ_pn_lo[::param.SpS], symbDelay)-θ)
plt.figure()
plt.plot(np.roll(ϕ_pn_lo[::param.SpS], symbDelay), label='phase of the LO');
plt.plot(θ+phaseOffSet, label='phase estimated by CPR');
plt.grid()
plt.xlim(0,θ.size)
plt.legend();

# correct (possible) phase ambiguity
rot = np.mean(symbTx[ind]/sigRx[ind])
sigRx  = rot*sigRx

# symbol normalization
sigRx = sigRx/np.sqrt(signal_power(sigRx[ind]))

# plot constellation after cpr
ax4.plot(sigRx[ind].real, sigRx[ind].imag,'.', markersize=4)
ax4.axis('square')
ax4.title.set_text('After CPR (DD-PLL)')
ax4.grid()

# estimate SNR of the received constellation
SNR = signal_power(symbTx[ind])/signal_power(sigRx[ind]-symbTx[ind])

# hard decision demodulation of the received symbols    
bitsRx = mod.demodulate(np.sqrt(mod.Es)*sigRx, demod_type = 'hard') 
bitsTx = mod.demodulate(np.sqrt(mod.Es)*symbTx, demod_type = 'hard') 

err = np.logical_xor(bitsRx[discard:bitsRx.size-discard], 
                     bitsTx[discard:bitsTx.size-discard])
BER = np.mean(err)

print('Estimated SNR = %.2f dB \n'%(10*np.log10(SNR)))
print('Total counted bits = %d  '%(err.size))
print('Total of counted errors = %d  '%(err.sum()))
print('BER = %.2e  '%(BER))

plt.figure()
plt.plot(err,'o', label = 'errors location')
plt.legend()
plt.grid()

# +
plt.figure(figsize=(5,5))
plt.ylabel('$S_Q$', fontsize=14)
plt.xlabel('$S_I$', fontsize=14)
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.grid()

plt.plot(sigRx[ind].real,sigRx[ind].imag,'.', markersize=4, label='Rx')
plt.plot(symbTx[ind].real,symbTx[ind].imag,'k.', markersize=4, label='Tx');

# +
from scipy.stats.kde import gaussian_kde

y = (sigRx[ind]).real
x = (sigRx[ind]).imag

k = gaussian_kde(np.vstack([x, y]))
k.set_bandwidth(bw_method=k.factor/4)

xi, yi = 1.1*np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.figure(figsize=(5,5))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1, shading='auto');


# -

def powerProfile(Pin, alpha, Lspan, Nspans):
    
    L = np.linspace(0, Nspans*Lspan, 2000)
    
    power = Pin-alpha*(L%Lspan)
    
    plt.plot(L, power,'')
    plt.xlabel('L [km]')
    plt.ylabel('power [dBm]')
    plt.title('Power profile')
    plt.grid()
    plt.xlim(min(L), max(L))


powerProfile(10, 0.2, 80, 10)

# ## Polarization multiplexed WDM signal

# **WDM signal generation**

# +
# Parâmetros do transmissor:
param = parameters()
param.M   = 16            # ordem do formato de modulação
param.Rs  = 32e9         # taxa de sinalização [baud]
param.SpS = 8            # número de amostras por símbolo
param.Nbits = 400000     # número de bits
param.pulse = 'rrc'      # formato de pulso
param.Ntaps = 4096       # número de coeficientes do filtro RRC
param.alphaRRC = 0.01    # rolloff do filtro RRC
param.Pch_dBm = 1        # potência média por canal WDM [dBm]
param.Nch     = 1        # número de canais WDM
param.Fc      = 193.1e12 # frequência central do espectro WDM
param.freqSpac = 40e9    # espaçamento em frequência da grade de canais WDM
param.Nmodes = 2         # número de modos de polarização

sigWDM_Tx, symbTx_, param = simpleWDMTx(param)

freqGrid = param.freqGrid
# -
# **Nonlinear fiber propagation with the split-step Fourier method**

# +
linearChannel = True

# optical channel parameters
paramCh = parameters()
paramCh.Ltotal = 80   # km
paramCh.Lspan  = 20    # km
paramCh.alpha = 0.2    # dB/km
paramCh.D = 16         # ps/nm/km
paramCh.Fc = 193.1e12  # Hz
paramCh.hz = 0.1       # km
paramCh.gamma = 1.3    # 1/(W.km)

if linearChannel:
    paramCh.hz = paramCh.Lspan 
    paramCh.gamma = 0  

Fs = param.Rs*param.SpS
sigWDM, paramCh = manakov_ssf(sigWDM_Tx, Fs, paramCh) 


receivedSignal = sigWDM.copy()
transmSymbols  = symbTx_.copy()
# -

# **Optical WDM spectrum before and after transmission**

# plot psd
plt.figure()
plt.xlim(paramCh.Fc-Fs/2,paramCh.Fc+Fs/2);
plt.psd(sigWDM_Tx[:,0], Fs=param.SpS*param.Rs, Fc=paramCh.Fc, NFFT = 4*1024, sides='twosided', label = 'WDM spectrum - Tx')
plt.psd(sigWDM[:,0], Fs=Fs, Fc=paramCh.Fc, NFFT = 4*1024, sides='twosided', label = 'WDM spectrum - Rx')
plt.legend(loc='lower left')
plt.title('optical WDM spectrum');


# **WDM channels coherent detection and demodulation**

# +
### Receiver

# parameters
chIndex  = 4    # index of the channel to be demodulated
polIndex = 0
plotPSD  = True

Fa = param.SpS*param.Rs
Fc = paramCh.Fc
Ta = 1/Fa
mod = QAMModem(m=param.M)

print('Demodulating channel #%d , fc: %.4f THz, λ: %.4f nm\n'\
      %(chIndex, (Fc + freqGrid[chIndex])/1e12, const.c/(Fc + freqGrid[chIndex])/1e-9))

sigWDM = receivedSignal[:,polIndex].reshape(len(sigWDM),)
symbTx = transmSymbols[:,polIndex,chIndex].reshape(len(symbTx_),)

# local oscillator (LO) parameters:
FO      = 64e6                 # frequency offset
Δf_lo   = freqGrid[chIndex]+FO  # downshift of the channel to be demodulated
lw      = 0*100e3                 # linewidth
Plo_dBm = 10                    # power in dBm
Plo     = 10**(Plo_dBm/10)*1e-3 # power in W
ϕ_lo    = 0                     # initial phase in rad    

print('Local oscillator P: %.2f dBm, lw: %.2f kHz, FO: %.2f MHz\n'\
      %(Plo_dBm, lw/1e3, FO/1e6))

# generate LO field
π       = np.pi
t       = np.arange(0, len(sigWDM))*Ta
ϕ_pn_lo = phaseNoise(lw, len(sigWDM), Ta)
sigLO   = np.sqrt(Plo)*np.exp(1j*(2*π*Δf_lo*t + ϕ_lo + ϕ_pn_lo))

# single-polarization coherent optical receiver
sigRx = coherentReceiver(sigWDM, sigLO)

# Rx filtering

# Matched filter
if param.pulse == 'nrz':
    pulse = pulseShape('nrz', param.SpS)
elif param.pulse == 'rrc':
    pulse = pulseShape('rrc', param.SpS, N=param.Ntaps, alpha=param.alphaRRC, Ts=1/param.Rs)

pulse = pulse/np.max(np.abs(pulse))            
sigRx = firFilter(pulse, sigRx)

# plot psd
if plotPSD:
    plt.figure();
   # plt.ylim(-250,-50);
    plt.psd(sigRx, Fs=Fa, NFFT = 16*1024, sides='twosided', label = 'Spectrum of the received signal')
    plt.legend(loc='upper left');
    plt.xlim(-Fa/2,Fa/2);

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(15,4.5))
fig.suptitle('Sequence of DSP blocks')

ax1.plot(sigRx.real, sigRx.imag,'.', markersize=4)
ax1.axis('square')
ax1.title.set_text('Output of coherent front-end')
ax1.grid()

# digital backpropagation
# hzDBP = 5
# Pin   = 10**(param.Pch_dBm/10)*1e-3
# sigRx = sigRx/np.sqrt(signal_power(sigRx))
# sigRx = dbp(np.sqrt(Pin)*sigRx, Fa, paramCh.Ltotal, paramCh.Lspan, hzDBP, paramCh.alpha, paramCh.gamma, paramCh.D, paramCh.Fc)
# sigRx = sigRx.reshape(len(sigRx),)
# sigRx = firFilter(pulse, sigRx)
    
# CD compensation
sigRx = edc(sigRx, paramCh.Ltotal, paramCh.D, Fc-Δf_lo, Fa)

# simple timing recovery
varVector = np.var((sigRx.T).reshape(-1,param.SpS), axis=0) # finds best sampling instant
sampDelay = np.where(varVector == np.amax(varVector))[0][0]

# downsampling
sigRx = sigRx[sampDelay::param.SpS]

discard = 1000
ind = np.arange(discard, sigRx.size-discard)

# symbol normalization
sigRx = sigRx/np.sqrt(signal_power(sigRx[ind]))

# plot constellation after CD compensation
ax2.plot(sigRx.real, sigRx.imag,'.', markersize=4)
ax2.axis('square')
ax2.title.set_text('After CD comp.')
ax2.grid()

# calculate time delay due to walkoff
symbDelay = np.argmax(signal.correlate(np.abs(symbTx), np.abs(sigRx)))-sigRx.size+1 

# compensate walkoff time delay
sigRx = np.roll(sigRx, symbDelay)

# symbol normalization
sigRx = sigRx/np.sqrt(signal_power(sigRx[ind]))

# estimate and compensate LO frequency offset
fo = fourthPowerFOE(sigRx, 1/param.Rs)
print('Estimated FO : %3.4f MHz'%(fo/1e6))

sigRx = sigRx*np.exp(-1j*2*π*fo*np.arange(0,len(sigRx))/param.Rs)

# plot constellation after LO frequency offset compensation
ax3.plot(sigRx[ind].real, sigRx[ind].imag,'.', markersize=4)
ax3.axis('square')
ax3.title.set_text('After CFR (4th-power FOE)')
ax3.grid()

# compensate phase noise (carrier phase recovery - cpr)
windowSize = 30
c  = mod.constellation/np.sqrt(mod.Es)
sigRx, ϕ, θ = cpr(sigRx, windowSize, c, symbTx)

# plot phases estimated by cpr
phaseOffSet = np.mean(np.roll(ϕ_pn_lo[::param.SpS], symbDelay)-θ)
plt.figure()
plt.plot(np.roll(ϕ_pn_lo[::param.SpS], symbDelay), label='phase of the LO');
plt.plot(θ+phaseOffSet, label='phase estimated by CPR');
plt.grid()
plt.xlim(0,θ.size)
plt.legend();

# correct (possible) phase ambiguity
rot = np.mean(symbTx[ind]/sigRx[ind])
sigRx  = rot*sigRx

# symbol normalization
sigRx = sigRx/np.sqrt(signal_power(sigRx[ind]))

# plot constellation after cpr
ax4.plot(sigRx[ind].real, sigRx[ind].imag,'.', markersize=4)
ax4.axis('square')
ax4.title.set_text('After CPR (DD-PLL)')
ax4.grid()

# estimate SNR of the received constellation
SNR = signal_power(symbTx[ind])/signal_power(sigRx[ind]-symbTx[ind])

# hard decision demodulation of the received symbols    
bitsRx = mod.demodulate(np.sqrt(mod.Es)*sigRx, demod_type = 'hard') 
bitsTx = mod.demodulate(np.sqrt(mod.Es)*symbTx, demod_type = 'hard') 

err = np.logical_xor(bitsRx[discard:bitsRx.size-discard], 
                     bitsTx[discard:bitsTx.size-discard])
BER = np.mean(err)

print('Estimated SNR = %.2f dB \n'%(10*np.log10(SNR)))
print('Total counted bits = %d  '%(err.size))
print('Total of counted errors = %d  '%(err.sum()))
print('BER = %.2e  '%(BER))

plt.figure()
plt.plot(err,'o', label = 'errors location')
plt.legend()
plt.grid()

# +
from scipy.stats.kde import gaussian_kde

y = (sigRx[ind]).real
x = (sigRx[ind]).imag

k = gaussian_kde(np.vstack([x, y]))
k.set_bandwidth(bw_method=k.factor/4)

xi, yi = 1.1*np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.figure(figsize=(5,5))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1, shading='auto');

# +
from numpy.matlib import repmat
from tqdm.notebook import tqdm

x = receivedSignal[::int(param.SpS/2)]
d = transmSymbols[:,:,0]

x = x.reshape(len(x),2)/np.sqrt(signal_power(x))
d = d.reshape(len(d),2)/np.sqrt(signal_power(d))

θ = np.pi/3

rot = np.array([[np.cos(θ), -np.sin(θ)],[np.sin(θ), np.cos(θ)]])

x = x@rot

plt.plot(x.real, x.imag,'.');
plt.plot(d.real, d.imag,'.');
plt.axis('square')
plt.xlim(-3.5, 3.5)
plt.ylim(-3.5, 3.5);

# +
from numba import njit, jit

def mimoAdaptEqualizer(x, dx=[], paramEq=[]):              
    
   
    # check input parameters
    numIter    = getattr(paramEq, 'numIter', 1)
    nTaps      = getattr(paramEq, 'nTaps', 15)
    mu         = getattr(paramEq, 'mu', 1e-3)
    SpS        = getattr(paramEq, 'SpS', 2)
    H          = getattr(paramEq, 'H', [])
    L          = getattr(paramEq, 'L', [])
    Hiter      = getattr(paramEq, 'Hiter', [])
    storeCoeff = getattr(paramEq, 'storeCoeff', False)
    alg        = getattr(paramEq, 'alg', 'nlms')
    M          = getattr(paramEq, 'M', 4)    
    
    # We want all the signal sequences to be disposed in columns:
    if not len(dx):
        dx = x.copy()
        
    try:
        if x.shape[1] > x.shape[0]:
            x = x.T       
    except IndexError:
        x  = x.reshape(len(x),1)       
        
    try:        
        if dx.shape[1] > dx.shape[0]:
            dx = dx.T
    except IndexError:        
        dx = dx.reshape(len(dx),1)

    nModes = int(x.shape[1]) # number of sinal modes (order of the MIMO equalizer)
    
    Lpad    = int(np.floor(nTaps/2))
    zeroPad = np.zeros((Lpad, nModes), dtype='complex')
    x = np.concatenate((zeroPad, x, zeroPad)) # pad start and end of the signal with zeros
    
    # Defining training parameters:
    mod = QAMModem(m=M) # commpy QAM constellation modem object
    constSymb = mod.constellation/np.sqrt(mod.Es) # complex-valued constellation symbols

    if not L:
        L = int(np.fix((len(x)-nTaps)/SpS+1)) # Length of the output (1 sample/symbol) of the training section       
    
    if len(H) == 0:
        H  = np.zeros((nModes**2, nTaps), dtype='complex')
        
        for initH in range(0, nModes):
            H[initH + initH*nModes, int(np.floor(H.shape[1]/2))] = 1 # Central spike initialization
           
    # Equalizer training:
    for indIter in tqdm(range(0, numIter)):
        print(alg,'training iteration #%d'%indIter)        
        yEq, H, errSq, Hiter = coreAdaptEq(x, dx, SpS, H, L, mu, nTaps, storeCoeff, alg, constSymb)               
        print(alg,'MSE = %.6f.'%np.nanmean(errSq))
        
    return  yEq, H, errSq, Hiter

@jit(nopython=True)
def coreAdaptEq(x, dx, SpS, H, L, mu, nTaps, storeCoeff, alg, constSymb):
    
    # allocate variables
    nModes  = int(x.shape[1])
    indTaps = np.arange(0, nTaps) 
    indMode = np.arange(0, nModes)
    
    errSq   = np.empty((nModes, L))
    yEq     = x[0:L].copy()    
    yEq[:]  = np.nan    
    outEq   = np.array([[0+1j*0]]).repeat(nModes).reshape(nModes, 1)
    
    if storeCoeff:
        Hiter = np.array([[0+1j*0]]).repeat((nModes**2)*nTaps*L).reshape(nModes**2, nTaps, L)
    else:
        Hiter = np.array([[0+1j*0]]).repeat((nModes**2)*nTaps).reshape(nModes**2, nTaps, 1)
        
    # Radii cma, rde
    Rcma = (np.mean(np.abs(constSymb)**4)/np.mean(np.abs(constSymb)**2))*np.ones((1, nModes))+1j*0          
    Rrde = np.unique(np.abs(constSymb))
        
    for ind in range(0, L):       
        outEq[:] = 0
            
        indIn = indTaps + ind*SpS # simplify indexing and improve speed

        # pass signal sequence through the equalizer:
        for N in range(0, nModes):
            inEq   = x[indIn, N].reshape(len(indIn), 1) # slice input coming from the Nth mode            
            outEq += H[indMode+N*nModes,:]@inEq         # add contribution from the Nth mode to the equalizer's output                 
                        
        yEq[ind,:] = outEq.T        
                    
        # update equalizer taps acording to adaptive specified 
        # training algorithm and save squared error:        
        if alg == 'nlms':
            H, errSq[:,ind] = nlmsUp(x[indIn, :], dx[ind,:], outEq, mu, H, nModes)
        elif alg == 'cma':
            H, errSq[:,ind] = cmaUp(x[indIn, :], Rcma, outEq, mu, H, nModes) 
        elif alg == 'ddlms':
            H, errSq[:,ind] = ddlmsUp(x[indIn, :], constSymb, outEq, mu, H, nModes)
        elif alg == 'rde':
            H, errSq[:,ind] = rdeUp(x[indIn, :], Rrde, outEq, mu, H, nModes)
        
        if storeCoeff:
            Hiter[:,:, ind] = H  
        else:
            Hiter[:,:, 1] = H
            
    return yEq, H, errSq, Hiter

@jit(nopython=True)
def nlmsUp(x, dx, outEq, mu, H, nModes):
    """
    coefficient update with the nlms algorithm    
    """          
    indMode = np.arange(0, nModes)    
    err = dx - outEq.T # calculate error w.r.t reference signal
    
    errDiag = np.diag(err[0]) # define diagonal matrix from error array
    
    # update equalizer taps according to nlms algorithm
    for N in range(0, nModes):
            indUpdTaps = indMode+N*nModes # simplify indexing and improve speed
            inAdapt = x[:, N].T/np.linalg.norm(x[:,N])**2 # NLMS normalization
            inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
            H[indUpdTaps,:] = H[indUpdTaps,:] + mu*errDiag@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2

@jit(nopython=True)
def ddlmsUp(x, constSymb, outEq, mu, H, nModes):
    """
    coefficient update with the ddlms algorithm    
    """      
    indMode    = np.arange(0, nModes)
    outEq      = outEq.T
    decided    = outEq.copy()
    decided[:] = np.nan
       
    for k in range(0, outEq.shape[1]):
        indSymb = np.argmin(np.abs(outEq[0,k] - constSymb))
        decided[0,k] = constSymb[indSymb]
                
    err = decided - outEq # calculate error w.r.t ddlms rule  

    errDiag = np.diag(err[0]) # define diagonal matrix from error array
   
    # update equalizer taps according to ddlms algorithm
    for N in range(0, nModes):
            indUpdTaps = indMode+N*nModes # simplify indexing
            inAdapt = x[:, N].T
            inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
            H[indUpdTaps,:] = H[indUpdTaps,:] + mu*errDiag@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2


@jit(nopython=True)
def cmaUp(x, R, outEq, mu, H, nModes):
    """
    coefficient update with the cma algorithm    
    """      
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    err   = R - np.abs(outEq)**2 # calculate error w.r.t cma rule  

    prodErrOut = np.diag(err[0])@np.diag(outEq[0]) # define diagonal matrix 
    
    # update equalizer taps according to cma algorithm
    for N in range(0, nModes):
            indUpdTaps = indMode+N*nModes # simplify indexing
            inAdapt = x[:, N].T
            inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
            H[indUpdTaps,:] = H[indUpdTaps,:] + mu*prodErrOut@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2

@jit(nopython=True)
def rdeUp(x, R, outEq, mu, H, nModes):
    """
    coefficient update with the rde algorithm    
    """      
    indMode    = np.arange(0, nModes)
    outEq      = outEq.T    
    decidedR    = outEq.copy()
    decidedR[:] = np.nan
    
    # find closest constellation radius
    for k in range(0, outEq.shape[1]):
        indR = np.argmin(np.abs(R - np.abs(outEq[0,k])))
        decidedR[0,k] = R[indR]
        
    err  = decidedR**2 - np.abs(outEq)**2 # calculate error w.r.t rde rule
    
    prodErrOut = np.diag(err[0])@np.diag(outEq[0]) # define diagonal matrix 
    
    # update equalizer taps according to rde algorithm
    for N in range(0, nModes):
            indUpdTaps = indMode+N*nModes # simplify indexing
            inAdapt = x[:, N].T
            inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
            H[indUpdTaps,:] = H[indUpdTaps,:] + mu*prodErrOut@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2


# +
paramEq = parameters()
paramEq.nTaps = 25
paramEq.SpS   = 2
paramEq.mu    = 2e-3
paramEq.numIter = 5
paramEq.storeCoeff = False
paramEq.alg   = 'nlms'
paramEq.M     = 16
#paramEq.H = H
#paramEq.L = 20000

y_EQ, H, errSq, Hiter = mimoAdaptEqualizer(x, dx=d, paramEq=paramEq)

# y_EQ, H, errSq, Hiter = mimoAdaptEqualizer(np.matlib.repmat(x,1,3),\
#                                            dx=np.matlib.repmat(d,1,3),\
#                                            paramEq=paramEq)

# +
fig, (ax1, ax2) = plt.subplots(1, 2)
discard = 1000

ax1.plot(y_EQ[discard:-discard,0].real, y_EQ[discard:-discard,0].imag,'.')
ax1.plot(d[:,0].real, d[:,0].imag,'.')
ax1.axis('square')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)

ax2.plot(y_EQ[discard:-discard,1].real, y_EQ[discard:-discard,1].imag,'.')
ax2.plot(d[:,1].real, d[:,1].imag,'.')
ax2.axis('square')
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)

plt.figure()
plt.plot(10*np.log10(errSq.T));

# +
plt.plot(H.real.T,'-');
plt.plot(H.imag.T,'-');

# plt.stem(H[0,:].real.T,linefmt='r');
# plt.stem(H[3,:].imag.T,linefmt='b');

# +
# #!pip install line_profiler
# -

# %load_ext line_profiler

# %lprun -f MIMO_NLMS_v1 MIMO_NLMS_v1(x, d, paramEq)

Hiter.shape

len(x.T)

x.shape[0]


