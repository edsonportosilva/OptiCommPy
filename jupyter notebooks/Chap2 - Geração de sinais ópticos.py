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

# +
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
# -

figsize(10, 3)

# # Geração de sinais ópticos
#
# Num sistema de comunicação digital óptica, a função do transmissor é converter uma dada sequência de bits num trem de pulsos elétricos que, por sua vez, será utilizado na modulação de uma portadora óptica (laser). A modulação de portadoras ópticas é realizada por meio de dispositivos de conversão eletro-óptica.
#
# Diversas técnicas de modulação podem ser implementadas e diversos fatores podem influenciar o projeto de um transmissor óptico. 

# +
Rs  = 10e9          # Taxa de símbolos (baud rate)
Ts  = 1/Rs          # Período de símbolo em segundos

t = np.arange(0, 11*Ts, Ts)/1e-12

for ind in range(0, 11):
    plt.vlines(t[ind], 0, 1, linestyles='dashed', color = 'k')
    
plt.xlabel('tempo [ps]');
plt.title('intervalos de sinalização');
plt.grid()
plt.xticks(t);
# -

# ## Formatos de modulação

# $$\begin{equation} \mathbf{E}(t)=\hat{\mathbf{e}} A \cos \left(\omega_c t + \phi\right) \end{equation}$$ em que $\omega_c = 2\pi f_{c}$ rad/s

# +
ϕ, omega_c, A, t = sp.symbols('ϕ, omega_c, A, t', real=True)

j = sp.I
π = sp.pi

E = A*cos(omega_c*t + ϕ)
E
# -

# $\mathbf{E}(t)=\operatorname{Re}\left[\hat{\mathbf{e}} A e^{j \phi} \exp \left(j \omega_c t\right)\right]$

sp.re(A*exp(j*ϕ)*exp(j*omega_c*t)).simplify()

sp.expand_trig(E).cancel()

# $$\begin{equation}
# \frac{\hat{E}_{\text {out }}(t)}{\hat{E}_{\text {in }}(t)}=\frac{1}{2} \left(e^{j \varphi_{1}(t)}+e^{j \varphi_{2}(t)}\right)
# \end{equation}$$

# $$
# \varphi_{1}(t)=\frac{u_{1}(t)}{V_{\pi_{1}}} \pi, \varphi_{2}(t)=\frac{u_{2}(t)}{V_{\pi_{2}}} \pi
# $$

# $$\begin{equation}
# \hat{E}_{\text {out}}(t)=\hat{E}_{\text {in }}(t)\cos \left(\frac{\Delta \varphi_{M Z M}(t)}{2}\right)=\hat{E}_{i n}(t) \cos \left(\frac{u(t)}{2 V_{\pi}} \pi\right)
# \end{equation}$$
#
# em que $\Delta \varphi_{M Z M}(t)=\varphi_{1}(t)-\varphi_{2}(t)=2 \varphi_{1}(t)$

# ### Chaveamento por deslocamento de amplitude (*amplitude shift-keying* - ASK) ou modulação de amplitude de pulso (*pulse amplitude modulation* - PAM)

# $ E(t)=\operatorname{Re}\left[A(t) e^{j \phi} \exp \left(j \omega_c t\right)\right]$

# $ A(t)= \sqrt{P_{0}} \left[ \sum_{n} b_{n} \delta \left(t-n T_{s}\right)\right] \ast p(t) = \sqrt{P_{0}} \sum_{n} b_{n} p\left(t-n T_{s}\right)$

from commpy.utilities  import signal_power, upsample
from utils.dsp import firFilter, pulseShape, eyediagram

# +
# parâmetros da simulação
SpS = 32

Rs     = 10e9          # Taxa de símbolos (para o caso do OOK Rs = Rb)
Ts     = 1/Rs          # Período de símbolo em segundos
Fa     = 1/(Ts/SpS)    # Frequência de amostragem do sinal (amostras/segundo)
Ta     = 1/Fa          # Período de amostragem
P0     = 1             # Potência


# +
# gera sequência de bits pseudo-aleatórios
bits   = np.random.randint(2, size=30)    
n      = np.arange(0, bits.size)

# mapeia bits para símbolos OOK
symbTx = np.sqrt(P0)*bits

plt.figure(1)
plt.stem(symbTx, basefmt=" ", use_line_collection=True)
plt.xlabel('n')
plt.ylabel('$b_n$')
plt.grid()
plt.xticks(np.arange(0, bits.size));

# +
# upsampling
symbolsUp = upsample(symbTx, SpS)

# pulso retangular ideal
pulse = pulseShape('rect', SpS)
pulse = pulse/max(abs(pulse))

t = np.arange(0, pulse.size)*(Ta/1e-12)

plt.figure(1)
plt.plot(t, pulse,'-', label = 'p(t)', linewidth=3)
plt.xlabel('tempo [ps]')
plt.ylabel('amplitude')
plt.xlim(min(t), max(t))
plt.grid()
plt.legend()

# formatação de pulso retangular
sigTx  = firFilter(pulse, symbolsUp)

# plota sinal 
t = np.arange(0, sigTx.size)*(Ta/1e-12)

# instantes centrais dos intervalos de sinalização
symbolsUp = upsample(2*bits-1, SpS)
symbolsUp[symbolsUp==0] = np.nan
symbolsUp = (symbolsUp + 1)/2

plt.figure(2)
plt.plot(t, sigTx.real,'-', linewidth=3)
plt.plot(t, symbolsUp.real, 'o')
plt.xlabel('tempo [ps]')
plt.ylabel('amplitude')
plt.title('$\sqrt{P_0}\; \sum_{n}b_{n}p(t-n T_s)$')
plt.grid()

t = (0.5*Ts + np.arange(0, bits.size*Ts, Ts))/1e-12
for ind in range(0, bits.size):
    plt.vlines(t[ind], 0, 1, linestyles='dashed', color = 'k')

# +
# pulso NRZ típico
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse))

t = np.arange(0, pulse.size)*(Ta/1e-12)

plt.figure(1)
plt.plot(t, pulse,'-', label = 'p(t)', linewidth=3)
plt.xlabel('tempo [ps]')
plt.ylabel('amplitude')
plt.xlim(min(t), max(t))
plt.grid()
plt.legend()

# upsampling
symbolsUp = upsample(symbTx, SpS)

# formatação de pulso retangular
sigTx  = firFilter(pulse, symbolsUp)

t = np.arange(0, sigTx.size)*(Ta/1e-12)

# instantes centrais dos intervalos de sinalização
symbolsUp = upsample(2*bits-1, SpS)
symbolsUp[symbolsUp==0] = np.nan
symbolsUp = (symbolsUp + 1)/2

plt.figure(2)
plt.plot(t, sigTx.real,'-',linewidth=3)
plt.plot(t, symbolsUp.real,'o')
plt.xlabel('tempo [ps]')
plt.ylabel('amplitude')
plt.title('$\sqrt{P_0}\; \sum_{n}b_{n}p(t-n T_s)$')
plt.grid()

t = (0.5*Ts + np.arange(0, bits.size*Ts, Ts))/1e-12
for ind in range(0, bits.size):
    plt.vlines(t[ind], 0, 1, linestyles='dashed', color = 'k')

# +
# pulso cosseno levantado (raised cosine)
Ncoeffs = 500
rolloff = 0.1

pulse = pulseShape('rc', SpS, Ncoeffs, rolloff, Ts)
pulse = pulse/max(abs(pulse))

t = np.arange(0, pulse.size)*(Ta/1e-12)

plt.figure(1)
plt.plot(t, pulse,'-', label = 'p(t)', linewidth=3)
plt.xlabel('tempo [ps]')
plt.ylabel('amplitude')
plt.xlim(min(t), max(t))
plt.grid()
plt.legend()

t = (-0.2*Ts + np.arange(0, (Ncoeffs/SpS)*Ts, Ts))/1e-12
for ind in range(0, t.size):
    #plt.vlines(t[ind], -0.2, 0.2, linestyles='dotted', color='r')
    plt.vlines(t[ind]+ 0.5*(Ts/1e-12), -0.2, 1, linestyles='dashed', color = 'k')

# upsampling
symbolsUp = upsample(symbTx, SpS)

# formatação de pulso 
sigTx  = firFilter(pulse, symbolsUp)

t = np.arange(0, sigTx.size)*(Ta/1e-12)

# instantes centrais dos intervalos de sinalização
symbolsUp = upsample(2*bits-1, SpS)
symbolsUp[symbolsUp==0] = np.nan
symbolsUp = (symbolsUp + 1)/2

plt.figure(2)
plt.plot(t, sigTx.real,'-', linewidth=3)
plt.plot(t, symbolsUp.real,'o')
plt.xlabel('tempo [ps]')
plt.ylabel('amplitude')
plt.title('$\sqrt{P_0}\; \sum_{n}b_{n}p(t-n T_s)$')
plt.grid()

t = (0.5*Ts + np.arange(0, bits.size*Ts, Ts))/1e-12
for ind in range(0, bits.size):
    plt.vlines(t[ind], 0, 1, linestyles='dashed', color = 'k')
# -

# ## Densidade espectral de potência do sinal modulado

# +
# gera sequência de bits pseudo-aleatórios
bits   = np.random.randint(2, size=10000)    
n      = np.arange(0, bits.size)

# mapeia bits para símbolos OOK
symbTx = bits
symbTx = np.sqrt(P0)*symbTx/np.sqrt(signal_power(symbTx))

# upsampling
symbolsUp = upsample(symbTx, SpS)

# pulso cosseno levantado (raised cosine)
Ncoeffs = 500
rolloff = 0.1

pulse = pulseShape('rc', SpS, Ncoeffs, rolloff, Ts)
pulse = pulse/max(abs(pulse))

# # pulso NRZ típico
# pulse = pulseShape('nrz', SpS)
# pulse = pulse/max(abs(pulse))

# formatação de pulso
sigTx  = firFilter(pulse, symbolsUp)

# plota psd
plt.figure();
plt.psd(sigTx,Fs=Fa, NFFT = 16*1024, sides='twosided', label = 'Espectro do sinal OOK')
plt.legend(loc='upper left');
plt.xlim(-3*Rs,3*Rs);
plt.ylim(-200,-50);

# +
Nsamples = 20000

# # diagrama de olho
# eyediagram(sigTx, Nsamples, SpS, n=3)
# -

# ### PAM4

# +
# gera sequência de bits pseudo-aleatórios
bits1   = np.random.randint(2, size=10000)  
bits2   = np.random.randint(2, size=10000) 

n      = np.arange(0, bits.size)

# mapeia bits para símbolos PAM4
symbTx = (2/3)*bits1 + (1/3)*bits2
symbTx = np.sqrt(P0)*symbTx/np.sqrt(signal_power(symbTx))

# upsampling
symbolsUp = upsample(symbTx, SpS)

# pulso NRZ típico
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse))

# formatação de pulso
sigTx  = firFilter(pulse, symbolsUp)

# plota psd
plt.figure();
plt.psd(sigTx,Fs=Fa, NFFT = 16*1024, sides='twosided', label = 'Espectro do sinal OOK')
plt.legend(loc='upper left');
plt.xlim(-3*Rs,3*Rs);
plt.ylim(-200,-50);

# +
Nsamples = 20000

# # diagrama de olho
# eyediagram(sigTx, Nsamples, SpS, n=3)
# -


