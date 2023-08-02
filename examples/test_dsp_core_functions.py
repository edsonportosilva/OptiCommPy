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

# <a href="https://colab.research.google.com/github/edsonportosilva/OptiCommPy/blob/main/examples/test_modulation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Test basic DSP functionalities

if 'google.colab' in str(get_ipython()):    
    # ! git clone -b main https://github.com/edsonportosilva/OptiCommPy
    from os import chdir as cd
    cd('/content/OptiCommPy/')
    # ! pip install . 

from optic.dsp.core import pnorm, signal_power, decimate, resample, lowPassFIR, firFilter, clockInterp
from optic.core import parameters
from optic.plot import eyediagram
import matplotlib.pyplot as plt
import numpy as np

# %load_ext autoreload
# %autoreload 2

# ## Test rational resample

# +
Fs = 1600
fc = 100

t = np.arange(0,4096)*(1/Fs)
π = np.pi

sig = np.sin(2*π*fc*t)

plt.plot(t, sig,'-o',markersize=4);
plt.xlim(min(t), max(t))

paramDec = parameters()
paramDec.SpS_in = 16
paramDec.SpS_out = 3
paramDec.Rs = fc

t_dec = resample(t, paramDec)
sig_dec = resample(sig, paramDec)
plt.plot(t_dec, sig_dec,'-o',markersize=4);
plt.xlim(0, 10*1/fc)
# -
# ## Test sampling clock converter

# +
Fs = 3200
fc = 100

t = np.arange(0, 40000)*(1/Fs)
π = np.pi

sig = np.sin(2*π*fc*t)

plt.plot(t, sig,'-o',markersize=4);
plt.xlim(min(t), max(t))


paramClk = parameters()
paramClk.Fs_in = 1600
paramClk.Fs_out = 1601
paramClk.AAF = False
paramClk.jitter_rms = 1e-8

t_dec = clockInterp(t, paramClk)
sig_dec = clockInterp(sig, paramClk)
plt.plot(t_dec, sig_dec,'-o',markersize=4);
plt.xlim(0, 10*1/fc)

eyediagram(sig_dec, sig_dec.size, Fs//fc, n=3, ptype='fast', plotlabel=None)
# -


