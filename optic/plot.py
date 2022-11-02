# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.kde import gaussian_kde

from optic.metrics import signal_power


def pconst(x, lim=False, R=1.5):
    """
    Plots signal constellations

    :param x: complex signals or list of complex signals

    """
    if type(x) == list:
        try:
            x[0].shape[1]
        except IndexError:
            x[0] = x[0].reshape(len(x[0]), 1)

        nSubPts = x[0].shape[1]
        radius = R * np.sqrt(signal_power(x[0]))
    else:
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
                    ax.plot(x[ind][:, k].real, x[ind][:, k].imag, ".")

                ax.axis("square")
                ax.grid()
                ax.set_title(f"mode {str(Position[k] - 1)}")

                if lim:
                    ax.set_xlim(-radius, radius)
                    ax.set_ylim(-radius, radius)
        else:
            for k in range(nSubPts):
                ax = fig.add_subplot(nRows, nCols, Position[k])
                ax.plot(x[:, k].real, x[:, k].imag, ".")
                ax.axis("square")
                ax.grid()
                ax.set_title(f"mode {str(Position[k] - 1)}")

                if lim:
                    ax.set_xlim(-radius, radius)
                    ax.set_ylim(-radius, radius)

        fig.tight_layout()

    elif nSubPts == 1:
        plt.figure()
        plt.plot(x.real, x.imag, ".")
        plt.axis("square")
        plt.grid()

        if lim:
            plt.xlim(-radius, radius)
            plt.ylim(-radius, radius)


    plt.show()

    return None


def eyediagram(sig, Nsamples, SpS, n=3, ptype="fast", plotlabel=None):
    """
    Plots the eye diagram of a modulated signal waveform

    :param Nsamples: number os samples to be plotted
    :param SpS: samples per symbol
    :param n: number of symbol periods
    :param type: 'fast' or 'fancy'
    :param plotlabel: label for the plot legend
    """

    if np.iscomplex(sig).any():
        d = 1
        plotlabel_ = f"{plotlabel} [real]"
    else:
        d = 0
        plotlabel_ = plotlabel

    for ind in range(d + 1):
        if ind == 0:
            y = sig[:Nsamples].real
            x = np.arange(0, y.size, 1) % (n * SpS)
        else:
            y = sig[:Nsamples].imag
            plotlabel_ = f"{plotlabel} [imag]"

        plt.figure()
        if ptype == "fancy":
            k = gaussian_kde(np.vstack([x, y]))
            k.set_bandwidth(bw_method=k.factor / 5)

            xi, yi = (
                1.1
                * np.mgrid[
                    x.min(): x.max(): x.size ** 0.5 * 1j,
                    y.min(): y.max(): y.size ** 0.5 * 1j,
                ]
            )
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1, shading="auto")
            plt.show()
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
