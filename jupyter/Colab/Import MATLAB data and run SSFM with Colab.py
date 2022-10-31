# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from os import chdir as cd
# ! git clone -b run-SSFM-Colab https://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@github.com/edsonportosilva/OptiCommPy-private
cd('/content/OptiCommPy-private')
# !pip install .
# !pip install numba --upgrade

from google.colab import drive
drive.mount('/content/drive')

# +
import matplotlib.pyplot as plt
import numpy as np

from optic.modelsGPU import manakovSSF
from optic.core import parameters
from optic.metrics import signal_power
from optic.plot import pconst

import scipy.constants as const

# +
import scipy.io

path = 'I:\\Meu Drive\\Colab\\Pesquisa\\Turbo equalização com PAS\\traces\\'


# Transmitter:
numberOfCarriers = 11;
M   = 256
Rs  = 32e9
Pin = -2;
WDMgrid  = 37.5e9;
pilotsOH = 0.05;
Mpilots  = 256;

# FEC parameters
codeBlocks = 18;
codeIndex  = 45;

# Channel
spans = 20;
spanLength = 50;

MonteCarloSim = 1;

traceID = f'SSF_{numberOfCarriers}xWDMCh_{int(Rs / 1000000000.0)}GBd_DP{M}QAM_{spans}x{spanLength}km_{codeBlocks}_blk_CI_{codeIndex}'


dataLoad = scipy.io.loadmat(path+traceID+'.mat')
# -

dataLoad['x']

# +
# optical channel parameters
paramCh = parameters()
paramCh.Ltotal = spans*spanLength         # total link distance [km]
paramCh.Lspan  = spanLength               # span length [km]
paramCh.alpha = dataLoad['alpha'][0][0]   # fiber loss parameter [dB/km]
paramCh.D = dataLoad['D'][0][0]           # fiber dispersion parameter [ps/nm/km]
paramCh.gamma = dataLoad['gamma'][0][0]   # fiber nonlinear parameter [1/(W.km)]
paramCh.Fc = dataLoad['Fc'][0][0]         # central optical frequency of the WDM spectrum
paramCh.hz = dataLoad['stepSize'][0][0]   # step-size of the split-step Fourier method [km]

Fs = dataLoad['Fs'][0][0] # sampling rate

# nonlinear signal propagation
sigWDM_Tx = np.array(dataLoad['x'], dataLoad['y'])
sigWDM, paramCh = manakovSSF(sigWDM_Tx, Fs, paramCh)
