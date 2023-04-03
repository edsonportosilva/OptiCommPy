# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
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

from optic.dsp import pulseShape, firFilter, decimate, symbolSync, pnorm
from optic.models import phaseNoise, pdmCoherentReceiver

try:
    from optic.modelsGPU import manakovSSF, manakovDBP
except:
    from optic.models import manakovSSF

from optic.tx import simpleWDMTx
from optic.core import parameters
from optic.equalization import edc, mimoAdaptEqualizer
from optic.carrierRecovery import cpr
from optic.metrics import fastBERcalc, monteCarloGMI, monteCarloMI, signal_power, calcEVM
from optic.plot import pconst, plotPSD

import scipy.constants as const

import logging as logg
logg.basicConfig(level=logg.INFO, format='%(message)s', force=True)

from copy import deepcopy

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
paramTx.M   = 64           # order of the modulation format
paramTx.Rs  = 32e9         # symbol rate [baud]
paramTx.SpS = 16           # samples per symbol
paramTx.pulse = 'rrc'      # pulse shaping filter
paramTx.Ntaps = 2*4096     # number of pulse shaping filter coefficients
paramTx.alphaRRC = 0.01    # RRC rolloff
paramTx.Pch_dBm = 0        # power per WDM channel [dBm]
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
paramCh.nlprMethod = True # use adaptive step-size based o maximum nonlinear phase-shift?
paramCh.maxNlinPhaseRot = 2e-2 # maximum nonlinear phase-shift per step
paramCh.prgsBar = True   # show progress bar?
#paramCh.saveSpanN = [1, 5, 9, 14]
Fs = paramTx.Rs*paramTx.SpS # sampling rate

# DBP parameters
paramDBP = deepcopy(paramCh)
paramDBP.nlprMethod = False
paramDBP.hz = 10
runDBP = True

### Receiver parameters

## LO parameters
FO      = 150e6                 # frequency offset
lw      = 100e3                 # linewidth
Plo_dBm = 10                    # power in dBm
Plo     = 10**(Plo_dBm/10)*1e-3 # power in W
ϕ_lo    = 0                     # initial phase in rad    

## photodiodes parameters
paramPD = parameters()
paramPD.B = paramTx.Rs
paramPD.Fs = Fs    
paramPD.ideal = True
    
Powers = paramTx.Pch_dBm + np.arange(-5,2,0.5)
scale = np.arange(-5,2,0.5)

BER = np.zeros((2,len(Powers)))
SER = np.zeros((2,len(Powers)))
MI  = np.zeros((2,len(Powers)))
GMI = np.zeros((2,len(Powers)))
NGMI = np.zeros((2,len(Powers)))
SNR = np.zeros((2,len(Powers)))
EVM = np.zeros((2,len(Powers)))

for indP, G in enumerate(scale):
    # nonlinear signal propagation
    G_lin = 10**(G/10)

    sigWDM, paramCh = manakovSSF(np.sqrt(G_lin)*sigWDM_Tx, Fs, paramCh)
    print('sigWDM power: ', round(10*np.log10(signal_power(sigWDM)/paramTx.Nch /1e-3),2),'dBm')
    ### WDM channels coherent detection and demodulation

    ### Receiver

    # parameters
    chIndex  = int(np.floor(paramTx.Nch/2))      # index of the channel to be demodulated

    Fc = paramCh.Fc
    Ts = 1/Fs
    freqGrid = paramTx.freqGrid

#     print('Demodulating channel #%d , fc: %.4f THz, λ: %.4f nm\n'\
#           %(chIndex, (Fc + freqGrid[chIndex])/1e12, const.c/(Fc + freqGrid[chIndex])/1e-9))

    symbTx = symbTx_[:,:,chIndex]

    #  set local oscillator (LO) parameters:   
    Δf_lo   = freqGrid[chIndex]+FO  # downshift of the channel to be demodulated

    # generate LO field
    π       = np.pi
    t       = np.arange(0, len(sigWDM))*Ts
    ϕ_pn_lo = phaseNoise(lw, len(sigWDM), Ts)
    sigLO   = np.sqrt(Plo)*np.exp(1j*(2*π*Δf_lo*t + ϕ_lo + ϕ_pn_lo))

    #### polarization multiplexed coherent optical receiver
    θsig = π/3 # polarization rotation angle
    sigRx = pdmCoherentReceiver(sigWDM, sigLO, θsig, paramPD)

    ### Matched filtering and CD compensation

    # Rx filtering
    
    # Matched filtering
    if paramTx.pulse == 'nrz':
        pulse = pulseShape('nrz', paramTx.SpS)
    elif paramTx.pulse == 'rrc':
        pulse = pulseShape('rrc', paramTx.SpS, N=paramTx.Ntaps, alpha=paramTx.alphaRRC, Ts=1/paramTx.Rs)

    pulse = pnorm(pulse)
    sigRx = firFilter(pulse, sigRx)

    # CD compensation/digital backpropagation
    if runDBP:
        Pch = 10**((G + paramTx.Pch_dBm)/10)*1e-3
        sigRx = np.sqrt(Pch/2)*pnorm(sigRx)
        print('channel input power (DBP): ', round(10*np.log10(signal_power(sigRx)/1e-3),3),'dBm')

        sigRx,_ = manakovDBP(sigRx, Fs, paramDBP)    
    else:
        sigRx = edc(sigRx, paramCh.Ltotal, paramCh.D, Fc-Δf_lo, Fs)

    ### Downsampling to 2 samples/symbol and re-synchronization with transmitted sequences

    # decimation
    paramDec = parameters()
    paramDec.SpS_in  = paramTx.SpS
    paramDec.SpS_out = 2
    sigRx = decimate(sigRx, paramDec)

    symbRx = symbolSync(sigRx, symbTx, 2)

    ### Power normalization

    x = pnorm(sigRx)
    d = pnorm(symbRx)

    ### Adaptive equalization

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

    y_EQ, H, errSq, Hiter = mimoAdaptEqualizer(x, dx=d, paramEq=paramEq)

    ### Carrier phase recovery

    paramCPR = parameters()
    paramCPR.alg = 'bps'
    paramCPR.M   = paramTx.M
    paramCPR.N   = 75
    paramCPR.B   = 64
    paramCPR.pilotInd = np.arange(0, len(y_EQ), 20) 

    y_CPR, θ = cpr(y_EQ, symbTx=d, paramCPR=paramCPR)

    y_CPR = pnorm(y_CPR)

    discard = 5000

    ### Evaluate transmission metrics

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


    BER[:,indP], SER[:,indP], SNR[:,indP] = fastBERcalc(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
    GMI[:,indP], NGMI[:,indP] = monteCarloGMI(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
    MI[:,indP] = monteCarloMI(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
    EVM[:,indP] = calcEVM(y_CPR[ind,:], paramTx.M, 'qam', d[ind,:])

    print('      pol.X      pol.Y      ')
    print(' SER: %.2e,  %.2e'%(SER[0,indP], SER[1,indP]))
    print(' BER: %.2e,  %.2e'%(BER[0,indP], BER[1,indP]))
    print(' SNR: %.2f dB,  %.2f dB'%(SNR[0,indP], SNR[1,indP]))
    print(' EVM: %.2f %%,    %.2f %%'%(EVM[0,indP]*100, EVM[1,indP]*100))
    print('  MI: %.2f bits, %.2f bits'%(MI[0,indP], MI[1,indP]))
    print(' GMI: %.2f bits, %.2f bits'%(GMI[0,indP], GMI[1,indP]))
    print('NGMI: %.2f,      %.2f'%(NGMI[0,indP], NGMI[1,indP]))
