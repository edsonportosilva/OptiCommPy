# %% [markdown]
# <a href="https://colab.research.google.com/github/edsonportosilva/OptiCommPy/blob/main/jupyter/basic_EDFA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Simulate a basic EDFA

# %%
if 'google.colab' in str(get_ipython()):    
    ! git clone -b main https://github.com/edsonportosilva/OptiCommPy
    from os import chdir as cd
    cd('/content/OptiCommPy/')
    ! pip install . 

# %%
import numpy as np
import matplotlib.pyplot as plt

from optic.core import parameters
from optic.tx import simpleWDMTx
from optic.amplification import edfaSM, OSA

from scipy.constants import c
from numpy.fft import fft,fftfreq

#import logging as logg
#logg.getLogger().setLevel(logg.INFO)
#logg.basicConfig(format='%(message)s')

# %%
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

# %%
figsize(10, 3)

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### EDFA parameters

# %%
# EDFA parameters
param_edfa = parameters()
# gain control
param_edfa.type     = "AGC"
param_edfa.value    = 20 #dB
# pid - gain control
param_edfa.kp = 1e-2
param_edfa.ki = 1e-2
param_edfa.kd = 5e-2
# pump information
param_edfa.forPump  = {'pump_signal': np.array([100e-3]), 'pump_lambda': np.array([980e-9])}
param_edfa.bckPump  = {'pump_signal': np.array([000e-3]), 'pump_lambda': np.array([980e-9])}
# giles parameters
param_edfa.file     = '..\\optic\\ampParams\\giles_MP980.dat'
param_edfa.fileunit = 'nm'
param_edfa.gmtc     = 'Bessel'
param_edfa.tol      = 0.05
param_edfa.tolCtrl  = 0.5


# %% [markdown]
# ### Transmitter parameters

# %%
# Transmitter parameters:
paramTx = parameters()
paramTx.M   = 4             # order of the modulation format
paramTx.Rs  = 40e9          # symbol rate [baud]
paramTx.SpS = 256           # samples per symbol
paramTx.Nbits    = 2**12       # total number of bits per polarization
paramTx.pulse    = 'rrc'       # pulse shaping filter
paramTx.Ntaps    = 1024        # number of pulse shaping filter coefficients
paramTx.alphaRRC = 0.01     # RRC rolloff
paramTx.Pch_dBm  = -15       # power per WDM channel [dBm]
paramTx.Nch      = 40        # number of WDM channels
paramTx.Fc       = c/1550e-9 # central optical frequency of the WDM spectrum
paramTx.freqSpac = 200e9    # WDM grid spacing
paramTx.Nmodes   = 2          # number of signal modes [2 for polarization multiplexed signals]

# %% [markdown]
# ### Generate WDM signal and amplification

# %%
# generate WDM signal
sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)

# %%
lenFrqSg,isy = np.shape(sigWDM_Tx)
Fs = paramTx.Rs*paramTx.SpS
simOpticalBand = (Fs*(c/paramTx.Fc)**2)/c
tx_pw = np.sum(1000*np.mean(sigWDM_Tx * np.conj(sigWDM_Tx), axis = 0).real)

# %%
# information TX
print('Sample rate [THz]: %5.3f' %(1e-12*Fs))
print('Time window [ns]:  %5.3f' %(1e9*lenFrqSg/Fs))
print('Central wavelength [nm]: %6.2f' %(1e9*c/paramTx.Fc))
print('Simulation window  [nm]: %f - [%6.2f nm - %6.2f nm]' 
      %(1e9*simOpticalBand, 1e9*(c/paramTx.Fc-simOpticalBand/2), 1e9*(c/paramTx.Fc+simOpticalBand/2)))
print('Frequency spacing [GHz]: %f' %(1e-9*Fs/lenFrqSg))
print('Number of points: %d' %(lenFrqSg))
print('Number of modes: %d' %(paramTx.Nmodes))
print('Average power - TX [mW] : %.3f mW' %(tx_pw))
print('Average power - TX [dBm] : %.3f dBm' %(10*np.log10(tx_pw)))

# %%
OSA(sigWDM_Tx, Fs, paramTx.Fc)

# %%
#%load_ext line_profiler
#%lprun -f edfaSM edfaSM(sigTxo, Fs, Fc, param_edfa)
# amplification
Eout, PumpF, PumpB = edfaSM(sigWDM_Tx, Fs, paramTx.Fc, param_edfa)

# %%
# information amp
rx_pw = np.sum(1000*np.mean(Eout * np.conj(Eout), axis = 0).real)
print('Forward pump  - [mW] : %.3f' %(1e3*PumpF[0]))
print('Backward pump - [mW] : %.3f' %(1e3*PumpB[1]))
print('Average power - RX amp [mW] : %.3f' %(rx_pw))
print('Average power - RX amp [dBm] : %.3f' %(10*np.log10(rx_pw)))
print('Gain [dB]: %.3f' %(10*np.log10(rx_pw/tx_pw)))

# %%
OSA(Eout, Fs, paramTx.Fc)

