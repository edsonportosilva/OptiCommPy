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

from os import chdir as cd
# ! git clone -b add-nonlinear-SDM-models https://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@github.com/edsonportosilva/OptiCommPy-private
cd('/content/OptiCommPy-private')
# !pip install .
# !pip install numba --upgrade

# +
import scipy.io

path = 'I:\\Meu Drive\\Colab\\Pesquisa\\Turbo equalização com PAS\\traces\\'


# Transmitter:
numberOfCarriers = 11;
M   = 256;        # Modulation format
Rs  = 32e9;       # Symbol rate

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

traceID = 'SSF_'+str(numberOfCarriers)+'xWDMCh_'+str(int(Rs/1e9))+\
            'GBd_DP'+str(M)+'QAM_'+str(spans)+'x'+str(spanLength)+'km_'+str(codeBlocks)+\
            '_blk_CI_'+str(codeIndex)

mat = scipy.io.loadmat(path+traceID+'.mat')
# -

mat


