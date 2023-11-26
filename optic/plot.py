"""Plot utilities."""
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_scatter_density
import numpy as np
import copy
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

from optic.dsp.core import pnorm, signal_power
import warnings

warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

def pconst(x, lim=True, R=1.25, pType="fancy", cmap="turbo", whiteb=True):
    """
    Plot signal constellations.
    
    Parameters
    ----------
    x : complex signals or list of complex signals
        Input signals.
    
    lim : bool, optional
        Flag indicating whether to limit the axes to the radius of the signal. 
        Defaults to True.
    
    R : float, optional
        Scaling factor for the radius of the signal. 
        Defaults to 1.25.
    
    pType : str, optional
        Type of plot. "fancy" for scatter_density plot, "fast" for fast plot.
        Defaults to "fancy".
    
    cmap : str, optional
        Color map for scatter_density plot.
        Defaults to "turbo".
    
    whiteb : bool, optional
        Flag indicating whether to use white background for scatter_density plot.
        Defaults to True.
    
    Returns
    -------
    fig : Figure
        Figure object.
    
    ax : Axes or array of Axes
        Axes object(s).
    
    """
    if type(x) == list:
        for ind, _ in enumerate(x):
            x[ind] = pnorm(x[ind])
        try:
            x[0].shape[1]
        except IndexError:
            x[0] = x[0].reshape(len(x[0]), 1)

        nSubPts = x[0].shape[1]
        radius = R * np.sqrt(signal_power(x[0]))
    else:
        x = pnorm(x)
        try:
            x.shape[1]
        except IndexError:
            x = x.reshape(len(x), 1)

        nSubPts = x.shape[1]
        radius = R * np.sqrt(signal_power(x))

    if nSubPts > 1:
        if nSubPts < 5:
            nCols = nSubPts
            nRows = 1
        elif nSubPts >= 6:
            nCols = int(np.ceil(nSubPts / 2))
            nRows = 2

        # Create a Position index
        Position = range(1, nSubPts + 1)

        fig = plt.figure()

        if type(x) == list:
            for k in range(nSubPts):           

                for ind in range(len(x)):
                    if pType == "fancy":
                        if ind == 0:
                            ax = fig.add_subplot(nRows, nCols, Position[k], projection='scatter_density')
                        ax = constHist(x[ind][:, k], ax, radius, cmap, whiteb)
                    elif pType == "fast":
                        if ind == 0:
                            ax = fig.add_subplot(nRows, nCols, Position[k])
                        ax.plot(x[ind][:, k].real, x[ind][:, k].imag, ".")

                ax.axis("square")
                ax.set_xlabel("In-Phase (I)")
                ax.set_ylabel("Quadrature (Q)")
                # ax.grid()
                ax.set_title(f"mode {str(Position[k] - 1)}")

                if lim:
                    ax.set_xlim(-radius, radius)
                    ax.set_ylim(-radius, radius)
        else:
            for k in range(nSubPts):                
                if pType == "fancy":
                    ax = fig.add_subplot(nRows, nCols, Position[k], projection='scatter_density')
                    ax = constHist(x[:, k], ax, radius, cmap, whiteb)
                elif pType == "fast":
                    ax = fig.add_subplot(nRows, nCols, Position[k])
                    ax.plot(x[:, k].real, x[:, k].imag, ".")

                ax.axis("square")
                ax.set_xlabel("In-Phase (I)")
                ax.set_ylabel("Quadrature (Q)")
                # ax.grid()
                ax.set_title(f"mode {str(Position[k] - 1)}")

                if lim:
                    ax.set_xlim(-radius, radius)
                    ax.set_ylim(-radius, radius)

        fig.tight_layout()

    elif nSubPts == 1:
        fig = plt.figure()
        #ax = plt.gca()
        if pType == "fancy":
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
            ax = constHist(x[:, 0], ax, radius, cmap, whiteb)
        elif pType == "fast":
            ax = plt.gca()
            ax.plot(x.real, x.imag, ".")
        plt.axis("square")
        ax.set_xlabel("In-Phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        # plt.grid()

        if lim:
            plt.xlim(-radius, radius)
            plt.ylim(-radius, radius)

    plt.show()

    return fig, ax


def constHist(symb, ax, radius, cmap="turbo", whiteb=True):
    """
    Generate histogram-based constellation plot.

    Parameters
    ----------
    symb : np.array
        Complex-valued constellation symbols.
    ax : axis object handle
        axis of the plot.
    radius : real scalar
        Parameter to adjust the x,y-range of the plot.

    Returns
    -------
    ax : axis object handle
        axis of the plot.

    """
    cmap = copy.copy(cm.get_cmap(cmap))
    if  whiteb:
        cmap.set_under(alpha=0)
    
    ax.scatter_density(symb.real, symb.imag, cmap=cmap, 
                             vmin=0.25, vmax=np.nanmax,
                             dpi=72, downres_factor=2)
    return ax


def eyediagram(sigIn, Nsamples, SpS, n=3, ptype="fast", plotlabel=None):
    """
    Plot the eye diagram of a modulated signal waveform.

    Parameters
    ----------
    sigIn : array-like
        Input signal waveform.
    Nsamples : int
        Number of samples to be plotted.
    SpS : int
        Samples per symbol.
    n : int, optional
        Number of symbol periods. Defaults to 3.
    ptype : str, optional
        Type of eye diagram. Can be 'fast' or 'fancy'. Defaults to 'fast'.
    plotlabel : str, optional
        Label for the plot legend. Defaults to None.

    Returns
    -------
    None
    """
    sig = sigIn.copy()

    if not plotlabel:
        plotlabel = " "

    if np.iscomplex(sig).any():
        d = 1
        plotlabel_ = f"{plotlabel} [real]" if plotlabel else "[real]"
    else:
        d = 0
        plotlabel_ = plotlabel

    for ind in range(d + 1):
        if ind == 0:
            y = sig[:Nsamples].real
            x = np.arange(0, y.size, 1) % (n * SpS)
        else:
            y = sig[:Nsamples].imag

            plotlabel_ = f"{plotlabel} [imag]" if plotlabel else "[imag]"
        plt.figure()
        if ptype == "fancy":
            f = interp1d(np.arange(y.size), y, kind="cubic")

            Nup = 40 * SpS
            tnew = np.arange(y.size) * (1 / Nup)
            y_ = f(tnew)

            taxis = (np.arange(y.size) % (n * SpS * Nup)) * (1 / Nup)
            imRange = np.array(
                [
                    [min(taxis), max(taxis)],
                    [min(y) - 0.1 * np.mean(np.abs(y)), 1.1 * max(y)],
                ]
            )

            H, xedges, yedges = np.histogram2d(
                taxis, y_, bins=350, range=imRange
            )

            H = H.T
            H = gaussian_filter(H, sigma=1.0)

            plt.imshow(
                H,
                cmap="turbo",
                origin="lower",
                aspect="auto",
                extent=[0, n, yedges[0], yedges[-1]],
            )

        elif ptype == "fast":
            y[x == n * SpS] = np.nan
            y[x == 0] = np.nan

            plt.plot(x / SpS, y, color="blue", alpha=0.8, label=plotlabel_)
            plt.xlim(min(x / SpS), max(x / SpS)) 

            if plotlabel is not None:
                plt.legend(loc="upper left")

        plt.xlabel("symbol period (Ts)")
        plt.ylabel("amplitude")
        plt.title(f"eye diagram {plotlabel_}")        
        plt.grid(alpha=0.15)
        plt.show()

    return None


def plotPSD(sig, Fs=1, Fc=0, NFFT=4096, fig=None, label=None):
    """
    Plot the power spectrum density (PSD) of a signal.

    Parameters
    ----------
    sig : np.array
        input signal.
    Fs : scalar, optional
         signal's sampling frequency. The default is 1.
    Fc : scalar, optional
        signal's central frequency. The default is 0.
    NFFT : scalar int, optional
        FFT size. The default is 4096.
    fig : figure object, optional
        matplotlib figure handle. The default is [].
    label : string, optional
        PSD plot label. The default is [].

    Returns
    -------
    fig : matplotlib figure object
        matplotlib figure object where the plot is generated.
    matplotlib axes object
        matplotlib axes object where the plot is displayed.

    """
    if fig is None:
        fig = []
    if label is None:
        label = []
    if not fig:
        fig = plt.figure()

    if not label:
        label = " "

    try:
       sig.shape[1]       
    except IndexError:
       sig = sig.reshape(len(sig), 1)

    for indMode in range(sig.shape[1]):
        plt.psd(
            sig[:, indMode],
            Fs=Fs,
            Fc=Fc,
            NFFT=NFFT,
            sides="twosided",
            label=f"{label}: Mode {str(indMode)}",
        )
    plt.legend(loc="lower left")
    plt.xlim(Fc - Fs / 2, Fc + Fs / 2)

    return fig, plt.gca()
