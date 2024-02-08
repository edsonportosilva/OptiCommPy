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

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/edsonportosilva/OptiCommPy/blob/main/examples/test_WDM_transmission.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="0270b2b0"
# # Simulation of coherent WDM transmission

# + colab={"base_uri": "https://localhost:8080/"} id="1ca1b9d6" outputId="842dd4fc-f03a-4069-cf21-3ff634969be9"
if 'google.colab' in str(get_ipython()):    
    # ! git clone -b main https://github.com/edsonportosilva/OptiCommPy
    from os import chdir as cd
    cd('/content/OptiCommPy/')
    # ! pip install . 

# + id="a4110d40"
import matplotlib.pyplot as plt
import numpy as np

from optic.dsp.core import pulseShape, firFilter, decimate, symbolSync, pnorm, signal_power
from optic.models.devices import pdmCoherentReceiver, basicLaserModel

try:
    from optic.models.modelsGPU import manakovSSF
except:
    from optic.models.channels import manakovSSF

from optic.models.tx import simpleWDMTx
from optic.utils import parameters, dBm2W
from optic.dsp.equalization import edc, mimoAdaptEqualizer
from optic.dsp.carrierRecovery import cpr
from optic.comm.metrics import fastBERcalc, monteCarloGMI, monteCarloMI, calcEVM
from optic.plot import pconst, plotPSD

import scipy.constants as const

import logging as logg
logg.basicConfig(level=logg.INFO, format='%(message)s', force=True)

# + colab={"base_uri": "https://localhost:8080/", "height": 17} id="7df01820" outputId="604d8ed4-041f-4280-ec2b-972c3a244a4d"
from IPython.core.display import HTML
from IPython.core.pylabtools import figsize

HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")

# + id="b8626f68"
figsize(10, 3)

# + id="fc09c144"
# %load_ext autoreload
# %autoreload 2
# #%load_ext line_profiler

# + [markdown] id="e22e32db"
#
# ## Transmitter

# + [markdown] id="f01da2ca"
# **Polarization multiplexed WDM signal generation**

# + colab={"base_uri": "https://localhost:8080/"} id="51257869" outputId="4efb007d-d5fe-4d7d-ad28-f0bbffdd13fd"
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

# generate WDM signal
sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)
# + [markdown] id="0cb851bf"
# **Nonlinear fiber propagation with the split-step Fourier method**

# + colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["d7ec57b1b19d4660a0548563dd43f97c", "f973387453444cc4b5fbec8658506a3a", "0d0d223577654c0980520ed48c4866a7", "92a22dde2b5e4ab882f824d5dff0d377", "7b1b87f7b77049a691df25723928eef3", "32ef48a5dd1d4a2cb94e5409dd572d74", "c5a9e5d034e64b00b295e93140f51e72", "b70bb6363ff64ccbb7087900ef892eb5", "d4247b94ef5c4a439cd4af9458125fc2", "6a786005faa04fd8b7c5e69dc70df06a", "994d17059a9b47f0b2ed3654712fb0c3"]} id="05599d49" outputId="debb83bb-27f7-46f4-be76-89fc838c11ac"
# optical channel parameters
paramCh = parameters()
paramCh.Ltotal = 700     # total link distance [km]
paramCh.Lspan  = 50      # span length [km]
paramCh.alpha = 0.2      # fiber loss parameter [dB/km]
paramCh.D = 16           # fiber dispersion parameter [ps/nm/km]
paramCh.gamma = 1.3      # fiber nonlinear parameter [1/(W.km)]
paramCh.Fc = paramTx.Fc  # central optical frequency of the WDM spectrum
paramCh.hz = 0.5         # step-size of the split-step Fourier method [km]
paramCh.maxIter = 5      # maximum number of convergence iterations per step
paramCh.tol = 1e-5       # error tolerance per step
paramCh.nlprMethod = True # use adaptive step-size based o maximum nonlinear phase-shift
paramCh.maxNlinPhaseRot = 2e-2 # maximum nonlinear phase-shift per step
paramCh.prgsBar = True   # show progress bar
#paramCh.saveSpanN = [1, 5, 9, 14]
paramCh.Fs = paramTx.Rs*paramTx.SpS # sampling rate

# nonlinear signal propagation
sigWDM = manakovSSF(sigWDM_Tx, paramCh)

# + [markdown] id="45da6d22"
# **Optical WDM spectrum before and after transmission**

# + colab={"base_uri": "https://localhost:8080/", "height": 241} id="489a01ea" outputId="a50a47a0-e564-4b88-de4d-21440eab470c"
# plot psd
Fs = paramCh.Fs
fig,_ = plotPSD(sigWDM_Tx, Fs, paramCh.Fc, label='Tx'); 
fig, ax = plotPSD(sigWDM, Fs, paramCh.Fc, fig=fig, label='Rx');
fig.set_figheight(3)
fig.set_figwidth(10)
ax.set_title('optical WDM spectrum');


# + [markdown] id="f291b19a"
# ### WDM channels coherent detection and demodulation

# + colab={"base_uri": "https://localhost:8080/", "height": 294} id="76945fb5" outputId="c57aaf50-d5cf-404b-cae2-60fb631b65e6"
# Receiver

# parameters
chIndex  = int(np.floor(paramTx.Nch/2))      # index of the channel to be demodulated

Fc = paramCh.Fc
Ts = 1/Fs
freqGrid = paramTx.freqGrid
π  = np.pi
t  = np.arange(0, len(sigWDM))*Ts

print('Demodulating channel #%d , fc: %.4f THz, λ: %.4f nm\n'\
      %(chIndex, (Fc + freqGrid[chIndex])/1e12, const.c/(Fc + freqGrid[chIndex])/1e-9))

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

# polarization multiplexed coherent optical receiver

# photodiodes parameters
paramPD = parameters()
paramPD.B = paramTx.Rs
paramPD.Fs = Fs    
paramPD.ideal = True

θsig = π/3 # polarization rotation angle
sigRx = pdmCoherentReceiver(sigWDM, sigLO, θsig, paramPD)

# plot received constellations
pconst(sigRx[0::paramTx.SpS,:], R=3);

# + [markdown] id="1cbf39db"
# ### Matched filtering and CD compensation

# + colab={"base_uri": "https://localhost:8080/", "height": 433} id="065d823a" outputId="15b896b0-0310-44b2-e3d1-269c74419fd4"
# Rx filtering

# Matched filtering
if paramTx.pulse == 'nrz':
    pulse = pulseShape('nrz', paramTx.SpS)
elif paramTx.pulse == 'rrc':
    pulse = pulseShape('rrc', paramTx.SpS, N=paramTx.Ntaps, alpha=paramTx.alphaRRC, Ts=1/paramTx.Rs)
    
pulse = pnorm(pulse)
sigRx = firFilter(pulse, sigRx)

# plot constellations after matched filtering
pconst(sigRx[0::paramTx.SpS,:], R=3)

# CD compensation
paramEDC = parameters()
paramEDC.L = paramCh.Ltotal
paramEDC.D = paramCh.D
paramEDC.Fc = Fc-Δf_lo
paramEDC.Fs = Fs

sigRx = edc(sigRx, paramEDC)

# plot constellations after CD compensation
pconst(sigRx[0::paramTx.SpS,:], R=3);

# + [markdown] id="901da914"
# ### Downsampling to 2 samples/symbol and re-synchronization with transmitted sequences

# + id="0d7e62a5"
# decimation
paramDec = parameters()
paramDec.SpS_in  = paramTx.SpS
paramDec.SpS_out = 2
sigRx = decimate(sigRx, paramDec)

symbRx = symbolSync(sigRx, symbTx, 2)

# + [markdown] id="e3813947"
# ### Power normalization

# + id="e6af9d14"
x = pnorm(sigRx)
d = pnorm(symbRx)

# + [markdown] id="f9c25025"
# ### Adaptive equalization

# + colab={"base_uri": "https://localhost:8080/", "height": 552, "referenced_widgets": ["305e5dc45df947c4b3640746643c4a26", "b07e3d1c0eee407ba3a2b6b3498edb61", "e0f8d22f269246b9b4ed8c6c7e61aeae", "5b6ae8306e70477580b6203d44e33a38", "0ef4e5276a4344638674b547b859a129", "0cec9d9b71ba45018460f4942672e73b", "596d425cdf7f4a0697238e3664c5195f", "9eef6ad9e86b4434b244d76fb90653b2", "d751e4cdab95433f8fabdaebf41d630a", "5773596ed660451aac75b1f0b2a052e5", "f6e0856c07ed4069940d749b31c5d6d5"]} id="512e12d6" outputId="e3f465e7-4243-45e7-90ec-cfd37a45d1c8"
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
    paramEq.alg = ['cma','cma'] # QPSK
    paramEq.mu = [5e-3, 1e-3] 
else:
    paramEq.alg = ['da-rde','rde'] # M-QAM
    paramEq.mu = [5e-3, 2e-4] 
    
y_EQ = mimoAdaptEqualizer(x, paramEq, d)

#plot constellations after adaptive equalization
discard = 5000
pconst(y_EQ[discard:-discard,:], R=1.5);

# + [markdown] id="aaf0f85c"
# ### Carrier phase recovery

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="4f6650fe" outputId="c5e97918-3305-4e71-8017-8b3432ff1e38"
paramCPR = parameters()
paramCPR.alg = 'bps'
paramCPR.M   = paramTx.M
paramCPR.N   = 75
paramCPR.B   = 64

y_CPR = cpr(y_EQ, paramCPR)

discard = 5000

#plot constellations after CPR
pconst(y_CPR[discard:-discard,:]);

# + [markdown] id="e9e07048"
# ### Evaluate transmission metrics

# + colab={"base_uri": "https://localhost:8080/"} id="67c66471" outputId="5e6538be-8488-470e-ab15-c1be2c1a9191"
ind = np.arange(discard, d.shape[0]-discard)

# remove phase and polarization ambiguities for QPSK signals
if paramTx.M == 4:   
    d = symbTx
    # find rotations after CPR and/or polarizations swaps possibly added at the output the adaptive equalizer:
    rot0 = [np.mean(pnorm(symbTx[ind,0])/pnorm(y_CPR[ind,0])), np.mean(pnorm(symbTx[ind,1])/pnorm(y_CPR[ind,0]))]
    rot1 = [np.mean(pnorm(symbTx[ind,1])/pnorm(y_CPR[ind,1])), np.mean(pnorm(symbTx[ind,0])/pnorm(y_CPR[ind,1]))]

    if np.argmax(np.abs(rot0)) == 1 and np.argmax(np.abs(rot1)) == 1:      
        y_CPR_ = y_CPR.copy() 
        # undo swap and rotation 
        y_CPR[:,0] = pnorm(rot1[np.argmax(np.abs(rot1))]*y_CPR_[:,1]) 
        y_CPR[:,1] = pnorm(rot0[np.argmax(np.abs(rot0))]*y_CPR_[:,0])
    else:
        # undo rotation
        y_CPR[:,0] = pnorm(rot0[np.argmax(np.abs(rot0))]*y_CPR[:,0])
        y_CPR[:,1] = pnorm(rot1[np.argmax(np.abs(rot1))]*y_CPR[:,1])
        

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
