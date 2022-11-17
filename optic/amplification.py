import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab   as mlab

from scipy import interpolate
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp

from scipy.constants import c, Planck
from scipy.special import jv, kv

import logging as logg
import copy

from numba import njit

def edfaSM(Ei, Fs, Fc, param_edfa):
    # Verify arguments
    param_edfa.type      = getattr(param_edfa, "type", "AGC")
    param_edfa.value     = getattr(param_edfa, "value", 20)
    param_edfa.nf        = getattr(param_edfa, "nf", 5)
    param_edfa.asep      = getattr(param_edfa, "asep", "X")
    param_edfa.file      = getattr(param_edfa, "file", "")
    param_edfa.fileunit  = getattr(param_edfa, "fileunit", "nm")
    param_edfa.a         = getattr(param_edfa, "a", 1.56e-6)
    param_edfa.b         = getattr(param_edfa, "b", 1.56e-6)
    param_edfa.rho       = getattr(param_edfa, "rho", 0.955e25)
    param_edfa.na        = getattr(param_edfa, "na", 0.22)
    param_edfa.gmtc      = getattr(param_edfa, "gmtc", "LP01")
    param_edfa.algo      = getattr(param_edfa, "algo", "Giles_spectrum")
    param_edfa.lngth     = getattr(param_edfa, "lngth", 8)
    param_edfa.tal       = getattr(param_edfa, "tal", 10e-3)
    param_edfa.forPump   = getattr(param_edfa, "forPump", {'pump_signal': np.array([100e-3]), 'pump_lambda': np.array([980e-9])})
    param_edfa.bckPump   = getattr(param_edfa, "bckPump", {'pump_signal': np.array([100e-3]), 'pump_lambda': np.array([980e-9])})
    param_edfa.lossS     = getattr(param_edfa, "lossS", 2.08 * 0.0001 * np.log10(10))
    param_edfa.lossP     = getattr(param_edfa, "lossP", 2.08 * 0.0001 * np.log10(10))
    param_edfa.longSteps = getattr(param_edfa, "longSteps", 100)
    param_edfa.tol       = getattr(param_edfa, "tol", 10/100)
    param_edfa.noiseBand = getattr(param_edfa, "noiseBand", 125e9)

    # Verify amplification type
    if (param_edfa.type not in ("AGC", "APC")):
        raise TypeError('edfaSM.type invalid argument.')

    # Verify giles file
    if not(os.path.exists(param_edfa.file)):
        raise TypeError('%s file doesn\'t exist.' %(param_edfa.file))

    # Verify algorithm argument
    if (param_edfa.algo not in ("Giles_spatial", "Giles_spectrum", "Saleh", "Jopson", "Inhomogeneous")):
        raise TypeError('edfaSM.algo invalid argument')

    fileT = np.loadtxt(param_edfa.file)
    # Verify file frequency unit
    if param_edfa.fileunit == "nm":
        lbFl = fileT[:,0] * 1e-9
    elif param_edfa.fileunit == "m":
        lbFl = fileT[:,0]
    elif param_edfa.fileunit == "Hz'":
        lbFl = c/fileT[:,0]
    elif param_edfa.fileunit == "THz":
        lbFl = c/fileT[:,0] * 1e-12
    else:
        raise TypeError('edfaSM.fileunit invalid argument - [nm - m - Hz - THz].')

    # Logitudinal step
    param_edfa.dr = param_edfa.a / param_edfa.longSteps
    param_edfa.r  = np.arange(0,param_edfa.a, param_edfa.dr)

    # Define field profile
    V = (2 * np.pi / lbFl) * param_edfa.a * param_edfa.na
    # u and v calculation for LP01 and Bessel profiles
    u = ((1 + np.sqrt(2)) * V) / (1 + (4 + V ** 4) ** 0.25)
    v = np.sqrt(V ** 2 - u ** 2)

    if param_edfa.gmtc == 'LP01':
        gamma = (((v * param_edfa.b) / (param_edfa.a * V * jv(1, u))) ** 2) * (jv(0, u * param_edfa.b / param_edfa.a) ** 2 + jv(1, u * param_edfa.b / param_edfa.a) ** 2)
        if param_edfa.algo == "Giles_spatial":
            print('Algo - gs')
            ##ik = 1 / np.pi * (v / (param_edfa.a * V) * jv(0, u * param_edfa.r / param_edfa.a) / jv(1, u)) ** 2
    else:
        if param_edfa.gmtc == 'Bessel':
            w_gauss = param_edfa.a * V / u * kv(1, v) / kv(0, v) *  jv(0, u)
        elif param_edfa.gmtc == 'Marcuse':
            w_gauss = param_edfa.a * (0.650 + 1.619 / V ** 1.5 + 2.879 / V ** 6)
        elif param_edfa.gmtc == 'Whitley':
            w_gauss = param_edfa.a * (0.616 + 1.660 / V ** 1.5 + 0.987 / V ** 6)
        elif param_edfa.gmtc == 'Desurvire':
            w_gauss = param_edfa.a * (0.759 + 1.289 / V ** 1.5 + 1.041 / V ** 6)
        elif param_edfa.gmtc == 'Myslinski':
            w_gauss = param_edfa.a * (0.761 + 1.237 / V ** 1.5 + 1.429 / V ** 6)
        else:
            raise TypeError('edfaSM.gmtc invalid argument - [LP01 - Marcuse - Whitley - Desurvire - Myslinski - Bessel].')
        gamma = 1 - np.exp(-2 * (param_edfa.b / w_gauss) ** 2)
        if param_edfa.algo == "Giles_spatial":
            print('Algo - gs')
            ##i_k = @(r) 2 ./ (pi .* w_gauss .^ 2) .* exp(-2 .* (r ./ w_gauss) .^ 2)
            ##i_k = cell2mat(arrayfun(@(x) i_k(x), properties.r, 'UniformOutput', false))

    if (np.sum(fileT[:,1]) > 1):
        logg.info("EDF absorption and gain coeficients. Calculating absorption and emission cross-section ...")
        absCoef  = 0.1 * np.log(10) * fileT[:,1]
        gainCoef = 0.1 * np.log(10) * fileT[:,2]
        # Doping profile is uniform with density RHO.
        absCross = absCoef  / param_edfa.rho / gamma
        emiCross = gainCoef / param_edfa.rho / gamma
    else:
        logg.info("EDF absorption and emission cross-section. Calculating absorption and gain coeficients ...")
        absCross = fileT[:,1]
        emiCross = fileT[:,2]
        # Doping profile is uniform with density RHO.
        absCoef  = param_edfa.absCross * param_edfa.rho * gamma
        gainCoef = param_edfa.emiCross * param_edfa.rho * gamma

    # Create second pol, if not exists
    lenFqSg, isy = np.shape(Ei)
    if (isy==1):
        Ei = np.concatenate((Ei, np.zeros(np.shape(Ei))), axis=1)
        isy += 1
    # Get signal in frequency domain
    freqSgn = Fs * fftfreq(len(Ei)) + Fc
    lenFqSg = len(freqSgn)

    # Get pump signal frequency points
    # TODO - more than one signal pump
    freqPmpFor = c / param_edfa.forPump['pump_lambda']
    freqPmpBck = c / param_edfa.bckPump['pump_lambda']
    pumpPmpFor = param_edfa.forPump['pump_signal']
    pumpPmpBck = param_edfa.bckPump['pump_signal']
    lenPmpFor  = np.size(freqPmpFor)
    lenPmpBck  = np.size(freqPmpBck)

    # Get optical band and specify frequency points for ASE calculation
    opticalBand = freqSgn.max() - freqSgn.min()
    freqASE = np.arange(-opticalBand/2, opticalBand/2, param_edfa.noiseBand) + Fc
    lenASE  = np.size(freqASE)

    #plt.plot(1e9*c/(freqSgn + Fc), 10*np.log10(1000*np.abs(fft(Ei)/lenFqSg)**2))

    # Define the frequency vector used in simulation.
    # SIGNALX + SIGNALY + FASEX + FASEY + FORPUMP
    param_edfa.freq = np.concatenate([freqSgn, freqSgn, freqASE, freqASE, freqPmpFor])
    param_edfa.ASE  = np.concatenate([np.zeros(isy*len(Ei)), np.ones(isy*lenASE), np.zeros(lenPmpFor)])
    param_edfa.uk   = np.ones(np.size(param_edfa.freq))
    # BCKPUMP + BASEX + BASEY
    freqAdd  = np.concatenate([freqPmpBck, freqASE, freqASE])
    ASEAdd   = np.concatenate([np.zeros(lenPmpBck), np.ones(isy*lenASE)])
    ukAdd    = np.concatenate([ np.ones(lenPmpBck), np.ones(isy*lenASE)])
    pInitAdd = np.concatenate([pumpPmpBck, np.zeros(isy*lenASE)])

    # Defines the optical paramters for each frequency to be used in spectral Giles algorithm.
    # SIGNALX + SIGNALY + FASEX + FASEY + FORPUMP
    param_edfa.absCoef  = np.interp(c / param_edfa.freq, lbFl,  absCoef)
    param_edfa.gainCoef = np.interp(c / param_edfa.freq, lbFl, gainCoef)

    #% BCKPUMP + BASEX + BASEY
    absCoefAdd  = np.interp(c / freqAdd, lbFl,  absCoef)
    gainCoefAdd = np.interp(c / freqAdd, lbFl, gainCoef)

    # Constantes numéricas para resolução das equações de taxa.
    xi = np.pi * param_edfa.b ** 2 * param_edfa.rho / param_edfa.tal
    param_edfa.const1 = (1 / (Planck * xi)) * (param_edfa.absCoef / param_edfa.freq)
    param_edfa.const2 = (1 / (Planck * xi)) * (param_edfa.absCoef + param_edfa.gainCoef) / param_edfa.freq
    param_edfa.const3 = (param_edfa.absCoef + param_edfa.gainCoef)
    param_edfa.const4 = (param_edfa.absCoef + param_edfa.lossS)
    param_edfa.const5 = (param_edfa.gainCoef * Planck * param_edfa.freq * param_edfa.noiseBand)

    # String para avaliação.
    evalStr = "solve_ivp(gilesSpectrum, zSpan, pInit, method='DOP853', args=(param_edfa,))"
    #evalStr = "odeint(gilesSpectrum, pInit, zSpan, args=(param_edfa,))"
    #evalStr = "solve_ivp(gilesSpectrum_f, zSpan, pInit, method='RK45', args=(param_edfa.const1, param_edfa.const2, param_edfa.const3, param_edfa.const4, param_edfa.const5, param_edfa.uk, param_edfa.ASE,))"
    #evalStr = "nbkode.RungeKutta45(gilesSpectrum, zSpan, pInit)"
    #evalStr = "lsoda(gilesSpectrum.address, zSpan, pInit, method='RK45', args=(param_edfa,))"
    

    ## Declara o sinal de potência do sinal.
    EiFt  = fft(Ei, axis = 0)
    Psgl  = np.reshape(np.abs(EiFt/lenFqSg)**2, (isy*lenFqSg), order='F')
    zSpan = np.array([0, param_edfa.lngth])
    #zSpan = np.linspace(0, param_edfa.lngth, 50)
    pInit = np.concatenate([Psgl, np.zeros(isy*lenASE), pumpPmpFor])

    # Propagação 0 -> L sem considerar BCKPUMP + BASE.
    sol = eval(evalStr)
    Pout = sol['y']
    zSpan = np.flip(zSpan)

    #plt.plot(1e9*c/freqSgn, 10*np.log10(1000*Eout[0:3200000,0]));plt.plot(1e9*c/freqSgn, 10*np.log10(1000*Eout[0:3200000,2]), alpha = 0.5)
    # Considera a ASE BACK e atualiza os parâmetros interpolados.
    # BCKPUMP + BASE
    pInit               = np.concatenate([pInit,                  pInitAdd])
    param_edfa.freq     = np.concatenate([param_edfa.freq,         freqAdd])
    param_edfa.ASE      = np.concatenate([param_edfa.ASE,           ASEAdd])
    param_edfa.uk       = np.concatenate([param_edfa.uk,            -ukAdd])
    param_edfa.absCoef  = np.concatenate([param_edfa.absCoef,   absCoefAdd])
    param_edfa.gainCoef = np.concatenate([param_edfa.gainCoef, gainCoefAdd])

    # Índices para atualização.
    idxPS  = np.arange(0,lenFqSg*isy)
    idxPAF = np.arange(idxPS[-1] +1, idxPS[-1]  + lenASE * isy + 1)
    idxPPF = np.arange(idxPAF[-1]+1, idxPAF[-1] + lenPmpFor + 1)
    idxPPB = np.arange(idxPPF[-1]+1, idxPPF[-1] + lenPmpBck + 1)
    idxPAB = np.arange(idxPPB[-1]+1, idxPPB[-1] + lenASE * isy + 1)

    # Constante utilizada para resolução das equações de taxa e propagação. Atualização.
    param_edfa.const1 = (1 / (Planck * xi)) * (param_edfa.absCoef / param_edfa.freq)
    param_edfa.const2 = (1 / (Planck * xi)) * (param_edfa.absCoef + param_edfa.gainCoef) / param_edfa.freq
    param_edfa.const3 = (param_edfa.absCoef + param_edfa.gainCoef)
    param_edfa.const4 = (param_edfa.absCoef + param_edfa.lossS)
    param_edfa.const5 = (param_edfa.gainCoef * Planck * param_edfa.freq * param_edfa.noiseBand) 

    # Controle de tentativas.
    tryLoop   = 0
    errorCvg  = 1
    MaxTry    = 15

    pInit[idxPS]  = Pout[idxPS,-1]
    pInit[idxPAF] = Pout[idxPAF,-1]
    pInit[idxPPF] = Pout[idxPPF,-1]

    while ((np.mean(np.abs(errorCvg)) > param_edfa.tol) and (tryLoop < MaxTry)):
        # Propagação L -> 0.
        sol = eval(evalStr)
        Pin = sol['y']
        zSpan = np.flip(zSpan)
        pInit = copy.deepcopy(Pin[:,-1])

        # Reinicia os valores de SIGNAL + FASE + FORPUMP.
        pInit[idxPS] = Psgl
        pInit[idxPAF] = np.zeros(lenASE*isy)
        pInit[idxPPF] = pumpPmpFor

        # Propagação 0 -> L.
        sol = eval(evalStr)
        Pout = sol['y']
        zSpan = np.flip(zSpan)
        pInit = copy.deepcopy(Pout[:,-1])
            
        # Reinicia os valores de BASE + BCKPUMP.
        pInit[idxPAB] = np.zeros(lenASE*isy)
        pInit[idxPPB] = pumpPmpBck
            
        # Verifica o critério de convergência. Considera os bombeios.
        if pumpPmpFor==0:
            errorCvg = 1 - Pout[idxPPB,-1] / pumpPmpBck
        elif pumpPmpBck==0:
            errorCvg = 1 - Pin[idxPPF,-1]  / pumpPmpFor
        else:
            errorCvg = 1 - (np.array([Pout[idxPPB,-1], Pin[idxPPF,-1]])) / np.array([pumpPmpBck, pumpPmpFor])
        print('EDFA: Laço %2d\n' %(tryLoop + 1))
        print('Convergência: %5.3f%%.\n' %(100 * np.mean(errorCvg)))
            
        # Caso não tenha convergido, determina o ponto médio e busca por passo.
        tryLoop = tryLoop + 1

    # Atualiza o sinal de saída.
    PpumpB = Pout[idxPPB, 0]
    PpumpF = Pout[idxPPF, -1]

    # Cria variáveis para ajustar o nível do ruído óptico.
    freqStep = Fs / lenFqSg
    resolutionOffSet  = param_edfa.noiseBand / freqStep

    # Atualiza o ruído óptico, independente da resolução do ruído gerado no EDFA.
    noiseB = Pout[idxPAB, 0]  / resolutionOffSet
    noiseF = Pout[idxPAF, -1] / resolutionOffSet

    # Interpola os valores do ruído óptico e adiciona a fase com distribuição  normal. É  necessário  dividir
    # por sqrt(2) os termos de ruído, pois na adição da fase, a amplitude deve ser unitária.

    f1_noiseb = interpolate.interp1d(freqASE, noiseB[0:lenASE], kind = 'linear', fill_value="extrapolate")
    f1_noisef = interpolate.interp1d(freqASE, noiseF[0:lenASE], kind = 'linear', fill_value="extrapolate")
    f2_noiseb = interpolate.interp1d(freqASE, noiseB[lenASE:],  kind = 'linear', fill_value="extrapolate")
    f2_noisef = interpolate.interp1d(freqASE, noiseF[lenASE:],  kind = 'linear', fill_value="extrapolate")
    
    noiseb = np.concatenate([np.sqrt(f1_noiseb(freqSgn), dtype = np.complex), np.sqrt(f2_noiseb(freqSgn), dtype = np.complex)])
    noisef = np.concatenate([np.sqrt(f1_noisef(freqSgn), dtype = np.complex), np.sqrt(f2_noisef(freqSgn), dtype = np.complex)])
    noiseF = noisef * (np.random.randn(lenFqSg*isy) + 1j * np.random.randn(lenFqSg*isy)) / np.sqrt(2)

    # Atualiza o sinal óptico e adiciona o ruído.
    # Atualiza o sinal óptico.
    Eout = np.reshape(np.sqrt(Pout[0:lenFqSg*isy, -1], dtype = np.complex), (lenFqSg, isy), order='F')
    Eout = Eout * np.exp(1j * np.angle(EiFt)) + np.reshape(noiseF, (lenFqSg, isy), order='F')
    Eout = ifft(Eout * lenFqSg, axis = 0)

    return Eout

def gilesSpectrum(z, P, properties):
    # Determina o número de portadores no nível metaestável.
    n2_normT1 = np.sum(P*properties.const1)
    n2_normT2 = np.sum(P*properties.const2) + 1
    n2_norm   = n2_normT1 / n2_normT2
    # Determina as matrices de termo de potência de sinal e potência de ASE.
    xi_k   = n2_norm * properties.const3 - properties.const4
    tauASE = n2_norm * properties.const5
    # Atualiza a variação de potência.
    return properties.uk * (P * xi_k + properties.ASE * tauASE)

#def gilesSpectrum(z, P, properties):
#    # Determina o número de portadores no nível metaestável.
#    n2_normT1 = dots(P,properties.const1)
#    n2_normT2 = dots(P,properties.const2) + 1
#    n2_norm   = n2_normT1 / n2_normT2
#    # Determina as matrices de termo de potência de sinal e potência de ASE.
#    xi_k   = n2_norm * properties.const3 - properties.const4
#    tauASE = n2_norm * properties.const5
#    # Atualiza a variação de potência.
#    return properties.uk * (P * xi_k + properties.ASE * tauASE)

@njit
def dots(x, y):
    s = 0
    for i in range(len(x)):
        s += x[i]*y[i]
    return s

@njit
def gilesSpectrum_f(z, P, const1, const2, const3, const4, const5, uk, ASE):
    # Determina o número de portadores no nível metaestável.
    n2_normT1 = np.sum(np.dot(P,const1))
    n2_normT2 = np.sum(np.dot(P,const2)) + 1
    n2_norm   = n2_normT1 / n2_normT2
    # Determina as matrices de termo de potência de sinal e potência de ASE.
    xi_k   = np.dot(n2_norm,const3) - const4
    tauASE = np.dot(n2_norm,const5)
    # Atualiza a variação de potência.
    temp1 = np.dot(P,xi_k)
    temp2 = np.dot(ASE,tauASE)
    return np.dot(uk,temp1 + temp2)

def fieldIntLP01(param_edfa, V):
    # u and v calculation
    u = ((1 + np.sqrt(2)) * V) / (1 + (4 + V ** 4) ** 0.25)
    v = np.sqrt(V ** 2 - u ** 2)
    gamma = (((v * param_edfa.b) / (param_edfa.a * V * jv(1, u))) ** 2) * (jv(0, u * param_edfa.b / param_edfa.a) ** 2 + jv(1, u * param_edfa.b / param_edfa.a) ** 2)
    ik    = 1 / np.pi * (v / (param_edfa.a * V) * jv(0, u * param_edfa.r / param_edfa.a) / jv(1, u)) ** 2 
    return gamma, ik

def OSA(Ei, Fs, Fc):
    spec, freqs = mlab.magnitude_spectrum(Ei, Fs=Fs, window = mlab.window_none, sides='twosided')
    freqs += Fc
    Z = 10 * np.log10(1000*(spec**2))
    line, = plt.plot(1e9*c/freqs, Z)
    plt.xlabel('Frequency [nm]')
    plt.ylabel('Magnitude [dBm]')
    plt.grid()
    return