# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from optic.metrics import signal_power


def pconst(x, lim=False, R=1.5):
    """
    Function to plot signal constellations

    :param x : complex signals or list of complex signals

    """
    if type(x) == list:
        nSubPts = x[0].shape[1]
        radius = R * np.sqrt(signal_power(x[0]))
    else:
        nSubPts = x.shape[1]
        radius = R * np.sqrt(signal_power(x))

    if nSubPts > 1:
        if nSubPts < 5:
            nCols = nSubPts
            nRows = 1
        elif nSubPts >= 6:
            nCols = np.ceil(nSubPts / 2)
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
                ax.set_title("mode " + str(Position[k] - 1))
                
                if lim:
                    ax.set_xlim(-radius, radius)
                    ax.set_ylim(-radius, radius)
        else:
            for k in range(nSubPts):
                ax = fig.add_subplot(nRows, nCols, Position[k])
                ax.plot(x[:, k].real, x[:, k].imag, ".")
                ax.axis("square")
                ax.grid()
                ax.set_title("mode " + str(Position[k] - 1))
                
                if lim:
                    ax.set_xlim(-radius, radius)
                    ax.set_ylim(-radius, radius)

    elif nSubPts == 1:
        plt.figure()
        plt.plot(x.real, x.imag, ".")
        plt.axis("square")
        
        if lim:
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)

    plt.show()
