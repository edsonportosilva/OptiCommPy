"""
=======================================================================
Customized functions for plotting and vizualization (:mod:`optic.plot`)
=======================================================================

.. autosummary::
   :toctree: generated/

   pconst                     -- Generate custom constellation plots
   constHist                  -- Generate histogram for constellation plots
   plotColoredConst           -- Colored constellation scatter plot
   plotDecisionBoundaries     -- Plot decision boundaries of the detector
   eyediagram                 -- Plots eyediagrams of communication signals
   plotPSD                    -- Plot power spectral density of signals
   randomCmap                 -- Generate a random RGB colormap
   animateConstGIF            -- Create and save a constellation plot animation as GIF
"""

"""Plot utilities."""
import copy
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_scatter_density  # noqa: F401  (registers the "scatter_density" projection)
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from optic.comm.modulation import detector
from optic.dsp.core import pnorm, signalPower
from optic.utils import dB2lin

# raised when scatter_density draws an empty density map (vmax=np.nanmax)
warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")


# ------------------------------------------------------------------ helpers


def _columns(x):
    """Signal as a 2D array of shape (N, nModes), with one column per mode."""
    x = np.asarray(x)

    return x.reshape(len(x), 1) if x.ndim == 1 else x


def _gridShape(nPlots):
    """Number of rows and columns of the subplots grid that holds `nPlots` plots."""
    if nPlots < 5:
        return 1, nPlots

    nRows = 2 if nPlots <= 10 else 3

    return nRows, int(np.ceil(nPlots / nRows))


def _colormap(cmap, transparentUnder=False):
    """Copy of a colormap (given by its name or object), optionally transparent below vmin."""
    cmap = mpl.colormaps.get_cmap(cmap)

    return cmap.with_extremes(under=(0, 0, 0, 0)) if transparentUnder else copy.copy(cmap)


def _figureAndAxes(fig, ax):
    """Figure and axes to be used by a plot: the given ones, or new ones if none is given."""
    if ax is None:
        if fig is None:
            return plt.subplots()
        return fig, fig.gca()

    return (ax.figure if fig is None else fig), ax


def _formatConstellationAxes(ax):
    """Square axes, with the labels of the in-phase and quadrature components."""
    ax.axis("square")
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")


# ----------------------------------------------------------- constellations


def pconst(x, lim=True, R=1.25, pType="fancy", cmap="turbo", whiteb=True, figsize=None):
    """
    Plot signal constellations.

    Parameters
    ----------
    x : complex signals or list of complex signals
        Input signals, of shape (N,) or (N, nModes). The signals of a list are
        plotted on top of each other (one subplot for each mode), and should have
        the same number of modes. The inputs are not modified.

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

    figsize : tuple, optional
        Figure size. If None, the default size of matplotlib is used.
        Defaults to None.

    Returns
    -------
    fig : Figure
        Figure object.

    ax : Axes
        Axes object of the plot (of the last mode, if there is more than one).

    Notes
    -----
    The power of each signal is normalized to one, and the modes are plotted in a grid
    of subplots (one row for up to 4 modes, two for up to 10 and three for more
    modes).
    """
    if pType not in ("fancy", "fast"):
        raise ValueError('pType must be either "fancy" or "fast"')

    signals = [_columns(pnorm(s)) for s in (x if isinstance(x, list) else [x])]
    nModes = signals[0].shape[1]
    radius = R * np.sqrt(signalPower(signals[0]))
    nRows, nCols = _gridShape(nModes)

    fig = plt.figure(figsize=figsize)

    for k in range(nModes):
        ax = fig.add_subplot(
            nRows, nCols, k + 1, projection="scatter_density" if pType == "fancy" else None
        )

        for signal in signals:
            if pType == "fancy":
                constHist(signal[:, k], ax, cmap, whiteb)
            else:
                ax.plot(signal[:, k].real, signal[:, k].imag, ".")

        _formatConstellationAxes(ax)

        if nModes > 1:
            ax.set_title(f"mode {k}")

        if lim:
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)

    if nModes > 1:
        fig.tight_layout()

    plt.show()
    plt.pause(0.01)  # Allow the plot to update

    return fig, ax


def constHist(symb, ax, cmap="turbo", whiteb=True):
    """
    Generate histogram-based constellation plot.

    Parameters
    ----------
    symb : np.array
        Complex-valued constellation symbols.
    ax : axis object handle
        axis of the plot, with the "scatter_density" projection.
    cmap : str or Colormap, optional
        Colormap name or object. The default is "turbo".
    whiteb : bool, optional
        If True, set values below the minimum to transparent (white background).
        The default is True.

    Returns
    -------
    ax : axis object handle
        axis of the plot.

    """
    ax.scatter_density(
        symb.real,
        symb.imag,
        cmap=_colormap(cmap, transparentUnder=whiteb),
        vmin=0.25,
        vmax=np.nanmax,
        dpi=72,
        downres_factor=2,
    )

    return ax


def plotColoredConst(
    symb,
    constSymb,
    px=None,
    SNR=20,
    rule="MAP",
    cmap="turbo",
    fig=None,
    ax=None,
):
    """
    Colored constellation scatter plot.

    Parameters
    ----------
    symb : np.array
        Complex-valued constellation symbols.
    constSymb : np.array
        Complex-valued constellation symbols used for detection.
    px : array_like, optional
        Prior probabilities of symbols.
    SNR : float, optional
        Signal-to-Noise Ratio (SNR) in decibels (dB). Default is 20 dB.
    rule : str, optional
        Detection rule, either "MAP" for Maximum A Posteriori or "ML" for Maximum Likelihood.
        Default is "MAP".
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for coloring the constellation symbols. Default is "turbo".
    fig : matplotlib.figure.Figure, optional
        Figure object for the plot. If None, the figure of `ax` is used, or a new
        figure is created if no axes is given. Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes object for the plot. If None, the current axes of `fig` is used, or a new
        axes is created if no figure is given. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for the plot.
    ax : matplotlib.axes.Axes
        Axes object for the plot.

    Notes
    -----
    This function generates a scatter plot of complex-valued constellation symbols with colors
    representing the corresponding decided constellation symbols based on detection results.

    The detected symbols are determined using a detector based on the provided input symbols, noise
    variance, detection rule, and prior probabilities (if available).
    """
    σ2 = 1 / dB2lin(SNR)

    _, pos = detector(symb, σ2, constSymb, rule=rule, px=px)  # detector

    # plot received symbols with colors that depend
    # on the respective decided constellation symbol
    colors = _colormap(cmap)(np.linspace(0, 1, len(constSymb)))

    fig, ax = _figureAndAxes(fig, ax)

    ax.scatter(symb.real, symb.imag, c=colors[pos], marker=".", s=0.5)
    _formatConstellationAxes(ax)
    plt.pause(0.01)  # Allow the plot to update

    return fig, ax


def plotDecisionBoundaries(
    constSymb,
    px=None,
    SNR=20,
    rule="MAP",
    gridStep=0.001,
    d=0.5,
    cmap="turbo",
    fig=None,
    ax=None,
):
    """
    Plot decision boundaries for a given constellation symbols.

    Parameters
    ----------
    constSymb : array_like
        An array of complex constellation symbols.
    px : array_like, optional
        Prior probabilities for each symbol in `constSymb`. If None, equal probabilities are assumed.
    SNR : float, optional
        Signal-to-noise ratio in decibels (dB). Default is 20.
    rule : str, optional
        The detection rule to use. Either 'MAP' (default) or 'ML'.
    gridStep : float, optional
        Step size for creating the decision boundary grid. Default is 0.001.
    d : float, optional
        Margin added to the maximum and minimum values of real and imaginary parts of `constSymb`.
        Default is 0.5.
    cmap : str or Colormap, optional
        Colormap to be used for the contour plot. Default is 'turbo'.
    fig : matplotlib.figure.Figure, optional
        Figure object for the plot. If None, the figure of `ax` is used, or a new
        figure is created if no axes is given. Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes object for the plot. If None, the current axes of `fig` is used, or a new
        axes is created if no figure is given. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    ax : matplotlib.axes.Axes
        The created matplotlib axes.

    Notes
    -----
    This function plots decision boundaries for a given set of constellation symbols in the complex plane.
    It uses the specified signal-to-noise ratio (SNR), detection rule, and prior probabilities (if available)
    to determine the decision boundaries.

    The decision boundaries are plotted using a contour plot with colors representing the different decision regions.
    """
    constSymb = pnorm(constSymb)  # normalize the constellation symbols

    if px is None:  # equal probabilities for all symbols
        px = np.full(len(constSymb), 1 / len(constSymb))

    # grid of received symbols that covers the constellation, with a margin d
    gI, gQ = np.meshgrid(
        np.arange(np.min(constSymb.real) - d, np.max(constSymb.real) + d, gridStep),
        np.arange(np.min(constSymb.imag) - d, np.max(constSymb.imag) + d, gridStep),
    )

    # decision regions of the detector for a Gaussian channel
    σ2 = 1 / dB2lin(SNR)
    _, pos = detector(gI.ravel() + 1j * gQ.ravel(), σ2, constSymb, rule=rule, px=px)

    fig, ax = _figureAndAxes(fig, ax)

    ax.contourf(gI, gQ, pos.reshape(gI.shape) + 1, 2 * len(constSymb), cmap=cmap)
    _formatConstellationAxes(ax)

    return fig, ax


# ---------------------------------------------------------------- eyediagram

# fraction of the peak density below which a bin of the 'fancy' eye diagram is empty
_EYE_EMPTY_DENSITY = 2e-2


def _eyeWaveforms(sigIn, Nsamples, label):
    """(waveform, label) of the parts to be plotted: real and, if complex, imaginary."""
    sig = np.asarray(sigIn[:Nsamples])

    if np.iscomplexobj(sig):
        parts = [(sig.real, f"{label} [real]"), (sig.imag, f"{label} [imag]")]
    else:
        parts = [(sig, label)]

    # float is required by the NaNs inserted in the lines of the "fast" eye diagram
    return [(np.asarray(part, dtype=np.float32), name.strip()) for part, name in parts]


def _eyeCoordinates(y, SpS, n):
    """Cubic interpolation of the waveform, and the time (in symbols, modulo n) of each point."""
    Nup = max(1, 1024 // SpS)  # upsampling factor for the interpolation
    yPlot = interp1d(np.arange(y.size), y, kind="cubic")(np.arange(y.size) / Nup)
    xPlot = (np.arange(y.size) % (n * SpS * Nup)) / (Nup * SpS)

    return xPlot, yPlot


def _eyeAxes(ax, nParts, dpi):
    """Axes of each eye diagram: the given ones (one per waveform part), or new ones."""
    if ax is None:
        return [plt.subplots(dpi=dpi)[1] for _ in range(nParts)]

    axesList = list(np.ravel(ax))
    if len(axesList) != nParts:
        raise ValueError(
            f"ax must hold {nParts} axes (one for each eye diagram), got {len(axesList)}"
        )

    return axesList


def _eyeHistogram(ax, xPlot, yPlot, SpS, n, cmap, bgColor):
    """Eye diagram as a smoothed 2D histogram of the interpolated waveform."""
    nsymb = yPlot.size // SpS
    if nsymb < 500000:  # tile the signal to ensure a high enough density
        reps = int(np.ceil(500000 / nsymb))
        xPlot, yPlot = np.tile(xPlot, reps), np.tile(yPlot, reps)

    yMargin = 0.1 * np.mean(np.abs(yPlot))
    imRange = [
        [np.min(xPlot), np.max(xPlot)],
        [np.min(yPlot) - yMargin, 1.1 * np.max(yPlot)],
    ]

    H, _, yEdges = np.histogram2d(xPlot, yPlot, bins=350, range=imRange)
    density = gaussian_filter(H.T, sigma=1.0)
    cmap = _colormap(cmap)

    if bgColor is not None:  # paint the bins without waveform with bgColor
        density = np.ma.masked_less(density, _EYE_EMPTY_DENSITY * density.max())
        cmap = cmap.with_extremes(bad=bgColor)

    ax.imshow(
        density,
        cmap=cmap,
        origin="lower",
        aspect="auto",
        extent=[0, n, yEdges[0], yEdges[-1]],
    )


def _eyeLines(ax, xPlot, yPlot, label, lineProps):
    """Eye diagram as the overlapped traces of the interpolated waveform."""
    # break the line where the time wraps around, so that it does not streak across
    yPlot[np.where(np.diff(xPlot) < 0)[0]] = np.nan

    props = {"color": "blue", "linewidth": 0.5, "alpha": 0.85, "label": label if label else None}
    props.update(lineProps or {})

    ax.plot(xPlot, yPlot, **props)
    ax.set_xlim(np.min(xPlot), np.max(xPlot))

    if props["label"]:
        ax.legend(loc="upper left")


def eyediagram(
    sigIn,
    Nsamples,
    SpS,
    n=3,
    ptype="fast",
    plotlabel="",
    dpi=None,
    cmap="turbo",
    bgColor=None,
    lineProps=None,
    ax=None,
    show=True,
):
    """
    Plot the eye diagram of a modulated signal waveform.

    Parameters
    ----------
    sigIn : array-like
        Input signal waveform. A complex signal is plotted as two eye diagrams, one
        for its real part and one for its imaginary part.
    Nsamples : int
        Number of samples of the signal to be used. See Notes.
    SpS : int
        Samples per symbol.
    n : int, optional
        Number of symbol periods. Defaults to 3.
    ptype : str, optional
        Type of eye diagram. Can be 'fast' or 'fancy'. Defaults to 'fast'.
    plotlabel : str, optional
        Label for the plot legend. Defaults to "".
    dpi : int, optional
        Dots per inch for the figure. If None, the default matplotlib DPI is used. Defaults to None.
    cmap : str or Colormap, optional
        Colormap of the density of the 'fancy' eye diagram. Defaults to 'turbo'.
    bgColor : color or None, optional
        Background color of the 'fancy' eye diagram, i.e., the color of the regions
        not crossed by the waveform. If None, they take the lowest color of `cmap`.
        Any matplotlib color is accepted, and 'none' plots the eye diagram without
        background (the regions are transparent and show the axes facecolor).
        Defaults to None.
    lineProps : dict, optional
        Properties of the traces of the 'fast' eye diagram, passed to
        ``Axes.plot`` (e.g. ``{"color": "k", "linewidth": 1, "linestyle": "--",
        "alpha": 0.3}``). They override the defaults ``color="blue"``,
        ``linewidth=0.5`` and ``alpha=0.85``. Defaults to None.
    ax : Axes or sequence of Axes, optional
        Axes where the eye diagrams are drawn, one for each eye diagram (two for a
        complex signal). If None, a new figure is created for each eye diagram.
        Defaults to None.
    show : bool, optional
        Whether to show the figures. Set it to False to draw or annotate on the
        eye diagrams before showing them. Defaults to True.

    Returns
    -------
    figList : list of Figure
        The figure of each eye diagram.
    axesList : list of Axes
        The axes of each eye diagram. They can be used to draw or annotate on top
        of the eye diagrams, in the time unit of the horizontal axis (symbol
        periods) and in the amplitude unit of the signal.

    Notes
    -----
    The waveform is interpolated (cubic) by the factor ``1024 // SpS``, and the eye
    diagram is made with `Nsamples` points of the interpolated waveform, which cover
    about ``Nsamples / 1024`` symbols. The 'fancy' eye diagram is a smoothed 2D
    histogram of the waveform, and the 'fast' one overlaps its traces.

    Examples
    --------
    Annotate the decision threshold and the sampling instant of an eye diagram:

    >>> figs, axes = eyediagram(sig, 50000, SpS, ptype="fancy", bgColor="none", show=False)
    >>> axes[0].axhline(0.5, color="w", linestyle="--")
    >>> axes[0].axvline(1.0, color="w", linestyle=":")
    >>> axes[0].annotate("threshold", xy=(0.1, 0.55), color="w")
    >>> plt.show()
    """
    if ptype not in ("fast", "fancy"):
        raise ValueError("ptype must be either 'fast' or 'fancy'")

    waveforms = _eyeWaveforms(sigIn, Nsamples, plotlabel.strip() if plotlabel else "")
    axesList = _eyeAxes(ax, len(waveforms), dpi)

    for (y, label), axEye in zip(waveforms, axesList):
        xPlot, yPlot = _eyeCoordinates(y, SpS, n)

        if ptype == "fancy":
            _eyeHistogram(axEye, xPlot, yPlot, SpS, n, cmap, bgColor)
        else:
            _eyeLines(axEye, xPlot, yPlot, label, lineProps)

        axEye.set_xlabel("Symbol period ($T_s$)")
        axEye.set_ylabel("Amplitude")
        axEye.set_title(label)
        axEye.grid(alpha=0.15)

    if show:
        plt.show(block=False)
        plt.pause(0.01)  # Allow the plot to update

    return [axEye.figure for axEye in axesList], axesList


# ------------------------------------------------------------ spectrum, GIF


def _fillBelow(ax, line, alpha):
    """Shade the area below a line, down to the bottom of the axes, without changing their limits."""
    x, y = line.get_xdata(), line.get_ydata()
    finite = np.isfinite(y)  # log10 of a null PSD is -inf
    if not finite.any():
        return

    # the shade goes far below the lowest level of the line, so that it always reaches
    # the bottom of the axes, but it is kept out of the data limits used for autoscaling
    floor = np.min(y[finite]) - 1e3
    dataLim, ylim = ax.dataLim.frozen(), ax.get_ylim()
    ax.fill_between(x, y, floor, where=finite, color=line.get_color(), alpha=alpha, linewidth=0)
    ax.dataLim.set(dataLim)
    ax.set_ylim(ylim, auto=None)  # keeps the limits (and ticks) set by Axes.psd, and autoscaling on


def plotPSD(sig, Fs=1, Fc=0, NFFT=4096, fig=None, label=None, fill=False, fillAlpha=0.25):
    """
    Plot the power spectrum density (PSD) of a signal.

    Parameters
    ----------
    sig : np.array
        input signal, of shape (N,) or (N, nModes).
    Fs : scalar, optional
         signal's sampling frequency. The default is 1.
    Fc : scalar, optional
        signal's central frequency. The default is 0.
    NFFT : scalar int, optional
        FFT size. The default is 4096.
    fig : figure object, optional
        matplotlib figure handle, to plot over an existing figure. The default is
        None, which creates a new figure.
    label : string, optional
        PSD plot label. If the signal has more than one mode (column), the label
        of each mode is ``"<label>: Mode <k>"``. The default is None (no legend).
    fill : bool, optional
        Whether to shade the area below the PSD line of each mode with the color of
        the line. The default is False.
    fillAlpha : float, optional
        Transparency of the shaded area, from 0 (transparent) to 1 (opaque). Only
        used if `fill` is True. The default is 0.25.

    Returns
    -------
    fig : matplotlib figure object
        matplotlib figure object where the plot is generated.
    matplotlib axes object
        matplotlib axes object where the plot is displayed.

    """
    if not fig:  # None (or the empty list used as default in previous versions)
        fig = plt.figure()

    ax = fig.gca()
    modes = _columns(sig).T

    for indMode, mode in enumerate(modes):
        if label is not None and len(modes) > 1:  # the modes are told apart in the legend
            modeLabel = f"{label}: Mode {indMode}"
        else:
            modeLabel = label

        *_, lines = ax.psd(
            mode,
            Fs=Fs,
            Fc=Fc,
            NFFT=NFFT,
            sides="twosided",
            label=modeLabel,
            return_line=True,
        )

        if fill:
            _fillBelow(ax, lines[0], fillAlpha)

    if label is not None:
        ax.legend(loc="lower left")
    ax.set_xlim(Fc - Fs / 2, Fc + Fs / 2)

    return fig, ax


def animateConstGIF(
    x,
    figName,
    xlabel="In-Phase (I)",
    ylabel="Quadrature (Q)",
    title=None,
    color="b",
    centralAxes=False,
    squareAxes=True,
    fram=200,
    inter=20,
    radius=2,
):
    """
    Create and save a constellation plot animation as GIF

    Parameters
    ----------
    x : numpy.ndarray
        Complex-valued signal to be animated.
    figName : str
        Figure file name with folder path.
    xlabel : str, optional
        X-axis label. Default is 'In-Phase (I)'.
    ylabel : str, optional
        Y-axis label. Default is 'Quadrature (Q)'.
    title : str, optional
        Title of the plot.
    color : str, optional
        Color of the points in the plot. Default is 'b' (blue).
    centralAxes : bool, optional
        Whether to place the axes at the center. Default is False.
    squareAxes : bool, optional
        Whether to keep the axes square. Default is True.
    fram : int, optional
        Number of frames. Default is 200.
    inter : int, optional
        Time interval between frames in milliseconds. Default is 20.
    radius : int, optional
        Radius for setting plot limits. Default is 2.

    Returns
    -------
    None
        The animation is saved to the file `figName`.

    Notes
    -----
    The animation is saved with ImageMagick, or with Pillow if ImageMagick is not
    available.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.grid()

    if squareAxes:
        ax.set_aspect("equal")

    if centralAxes:
        ax.spines["left"].set_position("center")
        ax.spines["bottom"].set_position("center")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)

    if title:
        ax.set_title(title)

    (line,) = ax.plot([], [], color + ".")

    period = max(1, int(len(x) / fram))  # number of symbols of each frame
    indx = np.arange(0, len(x), period)
    nFrames = min(fram, len(indx))

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        frame = x[indx[i] - period : indx[i]]
        line.set_data(frame.real, frame.imag)
        return (line,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=nFrames, interval=inter, blit=True
    )

    writer = "imagemagick" if animation.writers.is_available("imagemagick") else "pillow"
    anim.save(figName, dpi=200, writer=writer)
    plt.close(fig)


def randomCmap(nColors=100, low=0.1, high=0.99):
    """
    Generate a random colormap with the specified number of colors and random RGB values.

    Parameters
    ----------
    nColors : int, optional
        Number of colors in the colormap. Defaults to 100.
    low : float, optional
        Lower bound for random RGB values. Defaults to 0.1.
    high : float, optional
        Upper bound for random RGB values. Defaults to 0.99.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Random colormap with the specified number of colors and random RGB values.
    """
    return ListedColormap(np.random.uniform(low=low, high=high, size=(nColors, 3)), "new_map")
