"""Plot utilities."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

from optic.dsp import pnorm
from optic.metrics import signal_power


def pconst(x, lim=False, R=1.5, pType='fancy', cmap = 'turbo'):
    """
    Plot signal constellations.

    :param x: complex signals or list of complex signals

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
                ax = fig.add_subplot(nRows, nCols, Position[k])

                for ind in range(len(x)):
                    if pType == 'fancy':
                        ax = constHist(x[ind][:, k], ax, radius, cmap)
                    elif pType == 'fast':
                        ax.plot(x[ind][:, k].real, x[ind][:, k].imag, ".")

                ax.axis("square")
                ax.set_xlabel('In-Phase (I)')
                ax.set_ylabel('Quadrature (Q)')
                #ax.grid()
                ax.set_title(f"mode {str(Position[k] - 1)}")

                if lim:
                    ax.set_xlim(-radius, radius)
                    ax.set_ylim(-radius, radius)
        else:
            for k in range(nSubPts):
                ax = fig.add_subplot(nRows, nCols, Position[k])
                if pType == 'fancy':
                    ax = constHist(x[:, k], ax, radius, cmap)
                elif pType == 'fast':
                    ax.plot(x[:, k].real, x[:, k].imag, ".")

                ax.axis("square")
                ax.set_xlabel('In-Phase (I)')
                ax.set_ylabel('Quadrature (Q)')
                #ax.grid()
                ax.set_title(f"mode {str(Position[k] - 1)}")

                if lim:
                    ax.set_xlim(-radius, radius)
                    ax.set_ylim(-radius, radius)

        fig.tight_layout()

    elif nSubPts == 1:
        fig = plt.figure()
        ax = plt.gca()
        if pType == 'fancy':
            ax = constHist(x[:,0], ax, radius, cmap)
        elif pType == 'fast':
            ax.plot(x.real, x.imag, ".")       
        plt.axis("square")
        ax.set_xlabel('In-Phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        #plt.grid()

        if lim:
            plt.xlim(-radius, radius)
            plt.ylim(-radius, radius)

    plt.show()

    return fig, ax


def constHist(symb, ax, radius, cmap='turbo'):
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
    irange = radius*np.sqrt(signal_power(symb))
    imRange = np.array([[-irange, irange], [-irange, irange]])

    H, xedges, yedges = np.histogram2d(
        symb.real, symb.imag, bins=500, range=imRange
    )

    H = H.T
    
    H = gaussian_filter(H, sigma=8)
    ax.imshow(H, cmap=cmap, origin="lower", aspect="auto",
              extent=[-irange, irange, -irange, irange],
    )
    
    return ax


def eyediagram(sigIn, Nsamples, SpS, n=3, ptype="fast", plotlabel=None):
    """
    Plot the eye diagram of a modulated signal waveform.

    :param Nsamples: number os samples to be plotted
    :param SpS: samples per symbol
    :param n: number of symbol periods
    :param type: 'fast' or 'fancy'
    :param plotlabel: label for the plot legend
    """
    sig = sigIn.copy()

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

            Nup = 20*SpS
            tnew = np.arange(y.size) * (1 / Nup)
            y_ = f(tnew)

            taxis = (np.arange(y.size) % (n * SpS * Nup)) * (1 / Nup)
            imRange = np.array(
                [
                    [min(taxis), max(taxis)],
                    [min(y) - 0.1 * np.mean(y), 1.1 * max(y)],
                ]
            )

            H, xedges, yedges = np.histogram2d(
                taxis, y_, bins=350, range=imRange
            )

            H = H.T
            H = gaussian_filter(H, sigma=0.9)

            # plt.figure(figsize=(10, 3))
            plt.imshow(
                H,
                cmap='turbo',
                origin="lower",
                aspect="auto",
                extent=[0, n, yedges[0], yedges[-1]],
            )
            plt.xlabel("symbol period (Ts)")
            plt.ylabel("amplitude")
        elif ptype == "fast":
            y[x == n * SpS] = np.nan
            y[x == 0] = np.nan

            plt.plot(x / SpS, y, color="blue", alpha=0.8, label=plotlabel_)
            plt.xlim(min(x / SpS), max(x / SpS))
            plt.xlabel("symbol period (Ts)")
            plt.ylabel("amplitude")
            plt.title("eye diagram")

            if plotlabel is not None:
                plt.legend(loc="upper left")

            plt.grid()
            plt.show()
    return None
