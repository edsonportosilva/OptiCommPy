# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import cos, sin, exp

# # Geração de sinais ópticos
#
# Num sistema de comunicação digital óptica, a função do transmissor é converter uma dada sequência de bits num trem de pulsos elétricos que, por sua vez, será utilizado na modulação de uma portadora óptica (laser). A modulação de portadoras ópticas é realizada por meio de dispositivos de conversão eletro-óptica.
#
# Diversas técnicas de modulação podem ser implementadaas e diversos fatores podem influenciar o projeto de um transmissor óptico. 

# ## Formatos de modulação

# $\mathbf{E}(t)=\hat{\mathbf{e}} A_{0} \cos \left(\omega_{0} t-\phi_{0}\right)$

# +
ϕ0, omega0, A0, t = sp.symbols('ϕ0, omega0, A0, t', real=True)

j = sp.I

E = A0*cos(omega0*t - ϕ0)
E
# -

# $\mathbf{E}(t)=\operatorname{Re}\left[\hat{\mathbf{e}} A_{0} e^{j \phi_{0}} \exp \left(-j \omega_{0} t\right)\right]$

sp.re(A0*exp(j*ϕ0)*exp(-j*omega0*t)).simplify()

sp.expand_trig(E).cancel()

# ### Chaveamento por deslocamento de amplitude (*amplitude shift-keing* - ASK)

# $ E(t)=\operatorname{Re}\left[A_{0}(t) e^{j \phi_{0}} \exp \left(-j \omega_{0} t\right)\right]$

# $ A_{0}(t)=\sqrt{P_{0}} \sum_{n} b_{n} p\left(t-n T_{b}\right)$

# +
from commpy.modulation import Modem, QAMModem
from commpy.utilities  import signal_power, upsample
from commpy.filters    import rrcosfilter
from scipy.signal import lfilter

def filterNoDelay(h, x):
    """
    h: impulse response (symmetric)
    x: input signal 
    y: output signal
    """   
    N = h.size
    x = np.pad(x, (0, int(N/2)),'constant')
    y = lfilter(h,1,x)
    
    return y[int(N/2):y.size]

# def upsample(x, n):
    
#     x_up = np.zeros(x.size*n)   
#     x_up[0::n] = x
    
#     return x_up


# +
# parâmetros da simulação
SpS = 32

Rs     = 10e9          # Taxa de símbolos
Ts     = 1/Rs          # Período de símbolo em segundos
Fa     = 1/(Ts/SpS)    # Frequência de amostragem do sinal (amostras/segundo)
Ta     = 1/Fa          # Período de amostragem
P0     = 1             # Potência
    
# gera sequência de bits pseudo-aleatórios
bits   = np.random.randint(2, size=20)    
n      = np.arange(0, bits.size)

# mapeia bits para símbolos OOK
symbTx = np.sqrt(P0)*bits

plt.figure(1)
plt.stem(symbTx, use_line_collection=True)
plt.grid()

# +
# upsampling
symbolsUp = upsample(symbTx, SpS)

# pulso retangular
pulse = np.ones(int(SpS/2))

plt.figure(2)
plt.plot(pulse,'-')
plt.grid()

# formatação de pulso
sigTx  = filterNoDelay(pulse, symbolsUp)

t = np.arange(0, sigTx.size)*Ta

symbolsUp = upsample(2*bits-1, SpS)
symbolsUp[symbolsUp==0] = np.nan
symbolsUp = (symbolsUp + 1)/2

plt.figure(3)
plt.plot(t, sigTx,'-')
plt.plot(t, symbolsUp,'o')
plt.xlabel('tempo (s)')
plt.xlabel('amplitude (s)')
plt.grid()

# +
# gera sequência de bits pseudo-aleatórios
bits   = np.random.randint(2, size=10000)    
n      = np.arange(0, bits.size)

# mapeia bits para símbolos OOK
symbTx = np.sqrt(P0)*bits

# upsampling
symbolsUp = upsample(symbTx, SpS)

# pulso retangular
pulse = np.ones(int(SpS/2))

# formatação de pulso
sigTx  = filterNoDelay(pulse, symbolsUp)

# plot spectrums
plt.figure();
plt.psd(sigTx,Fs=Fa, NFFT = 16*1024, sides='twosided', label = 'Tx spectrum')
plt.legend(loc='upper left');
plt.xlim(-8*Rs,8*Rs);
# -


