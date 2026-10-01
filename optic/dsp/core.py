"""
================================================================
Core digital signal processing utilities (:mod:`optic.dsp.core`)
================================================================

.. autosummary::
   :toctree: generated/

   sigPow                  -- Calculate the average power of x.
   signalPower             -- Calculate the total average power of x.
   firFilter               -- Perform FIR filtering and compensate for filter delay.
   rrcFilterTaps           -- Generate Root-Raised Cosine (RRC) filter coefficients.
   rcFilterTaps            -- Generate Raised Cosine (RC) filter coefficients.
   pulseShape              -- Generate a pulse shaping filter.
   clockSamplingInterp     -- Interpolate signal to a given sampling rate.
   quantizer               -- Quantize the input signal using a uniform quantizer.
   lowPassFIR              -- Calculate FIR coefficients of a lowpass filter.
   decimate                -- Decimate signal.
   resample                -- Signal resampling.
   upsample                -- Upsample a signal by inserting zeros between samples.
   symbolSync              -- Synchronizer delayed sequences of symbols.
   finddelay               -- Estimate the delay between sequences of symbols.
   pnorm                   -- Normalize the average power of each componennt of x.
   anorm                   -- Normalize the amplitude of each componennt of x.
   gaussianComplexNoise    -- Generate complex-valued circular Gaussian noise.
   gaussianNoise           -- Generate Gaussian noise.
   phaseNoise              -- Generate realization of a random-walk phase-noise process.
   movingAverage           -- Calculate the sliding window moving average.
   delaySignal             -- Apply a time delay to a signal.
   blockwiseFFTConv        -- Calculates convolutions in the frequency domain.
   freqShift               -- Applies a frequency shift to a signal.
   iqMixing                -- Add IQ mixing (IQ imbalance and skew) to a signal.
   calcMZM                 -- Fast function to be used in the Mach-Zehnder modulator (MZM) model.
   calcPM                  -- Fast function to be used in the phase modulator (PM) model.
   levinson                -- Solve the Toeplitz system of equations using the Levinson-Durbin algorithm.
   autocorr                -- Estimate the autocorrelation coefficients of a signal.
   estimateWhiteningFilter -- Estimate the coefficients of a whitening filter using the autocorrelation method.
"""

"""Digital signal processing utilities."""
import logging as logg

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift, ifft

from optic.utils import parameters


@njit
def sigPow(x):
    r"""
    Calculate the average power of x per mode.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    scalar
        Average power of x: P = mean(abs(x)**2).

    Notes
    -----
    For a discrete-time signal :math:`x[n]`, :math:`n = 0, \ldots, N-1`, the average
    power is the mean squared magnitude of its samples,

    .. math::
        P_x = \frac{1}{N}\sum_{n=0}^{N-1} |x[n]|^2. \tag{1}

    If :math:`x` has several columns, the average in Eq. (1) is taken over all of its
    elements.
    """
    return np.mean(np.abs(x) ** 2)


def signalPower(x):
    r"""
    Calculate the total power of x.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    scalar
        Total power of x: P = sum(abs(x)**2).

    Notes
    -----
    Let :math:`x_k[n]`, :math:`k = 1, \ldots, K`, denote the :math:`K` components (columns)
    of the signal, e.g. the polarization or spatial modes of an optical field. The total
    power is the sum of the average powers of the components,

    .. math::
        P = \sum_{k=1}^{K} \frac{1}{N}\sum_{n=0}^{N-1} |x_k[n]|^2. \tag{1}
    """
    return np.sum(np.mean(x * np.conj(x), axis=0).real)


def firFilter(h, x):
    r"""
    Perform FIR filtering and compensate for filter delay.

    Parameters
    ----------
    h : np.array
        Coefficients of the FIR filter (impulse response, symmetric).
    x : np.array
        Input signal.

    Returns
    -------
    y : np.array
        Output (filtered) signal.

    Notes
    -----
    The output of a finite impulse response (FIR) filter with coefficients
    :math:`h[k]`, :math:`k = 0, \ldots, L-1`, is given by the discrete convolution
    :math:`(h * x)[n] = \sum_{k} h[k]\, x[n-k]`. Since a filter with symmetric
    coefficients delays the signal by :math:`D = \lfloor (L-1)/2 \rfloor` samples, this
    delay is removed from the output, which is computed as

    .. math::
        y[n] = \sum_{k=0}^{L-1} h[k]\, x[n + D - k], \qquad n = 0, \ldots, N-1, \tag{1}

    so that :math:`y[n]` has the same length as :math:`x[n]` and is time-aligned with
    it. The convolution is evaluated with the overlap-add method, which is efficient
    when :math:`L \ll N`. Each column of :math:`x` is filtered independently.

    References
    ----------
    [1] P. S. R. Diniz, E. A. B. da Silva, e S. L. Netto, Digital Signal Processing: System Analysis and Design. Cambridge University Press, 2010.
    """
    try:
        x.shape[1]
        input1D = False
    except IndexError:
        input1D = True
        # If x is a 1D array, reshape it to a 2D array with one column
        x = x.reshape(len(x), 1)

    y = x.copy()
    nModes = x.shape[1]

    for n in range(nModes):
        # overlap-add convolution: faster than a single long FFT when len(h) << len(x)
        y[:, n] = signal.oaconvolve(x[:, n], h, mode="same")

    if input1D:
        # If the input is 1D, return it as a 1D array
        y = y.flatten()

    return y


@njit
def rrcFilterTaps(t, alpha, Ts):
    r"""
    Generate Root-Raised Cosine (RRC) filter coefficients.

    Parameters
    ----------
    t : np.array
        Time values.
    alpha : float
        RRC roll-off factor.
    Ts : float
        Symbol period.

    Returns
    -------
    coeffs : np.array
        RRC filter coefficients.

    Notes
    -----
    The root-raised cosine (RRC) pulse with symbol period :math:`T_s` and roll-off
    factor :math:`0 \le \alpha \le 1` is the pulse whose squared magnitude spectrum is
    the raised cosine spectrum, so that a matched pair of RRC filters (transmitter and
    receiver) produces a Nyquist pulse, free of intersymbol interference. Its impulse
    response is

    .. math::
        p(t) = \frac{1}{T_s}\,
        \frac{\sin\!\left[\pi \frac{t}{T_s}(1-\alpha)\right]
        + 4\alpha\frac{t}{T_s}\cos\!\left[\pi \frac{t}{T_s}(1+\alpha)\right]}
        {\pi \frac{t}{T_s}\left[1-\left(4\alpha\frac{t}{T_s}\right)^2\right]}, \tag{1}

    with the limiting values

    .. math::
        p(0) = \frac{1}{T_s}\left[1 + \alpha\left(\frac{4}{\pi}-1\right)\right], \tag{2}

    .. math::
        p\left(\pm\frac{T_s}{4\alpha}\right) = \frac{\alpha}{T_s\sqrt{2}}
        \left[\left(1+\frac{2}{\pi}\right)\sin\frac{\pi}{4\alpha}
        + \left(1-\frac{2}{\pi}\right)\cos\frac{\pi}{4\alpha}\right]. \tag{3}

    The bandwidth occupied by the pulse is :math:`(1+\alpha)/(2T_s)`.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.
    """
    coeffs = np.zeros(len(t), dtype=np.float64)

    for i, t_i in enumerate(t):
        t_abs = abs(t_i)
        if t_i == 0:
            coeffs[i] = (1 / Ts) * (1 + alpha * (4 / np.pi - 1))
        elif t_abs == Ts / (4 * alpha):
            term1 = (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
            term2 = (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            coeffs[i] = (alpha / (Ts * np.sqrt(2))) * (term1 + term2)
        else:
            t1 = np.pi * t_i / Ts
            t2 = 4 * alpha * t_i / Ts
            coeffs[i] = (
                (1 / Ts)
                * (
                    np.sin(t1 * (1 - alpha))
                    + 4 * alpha * t_i / Ts * np.cos(t1 * (1 + alpha))
                )
                / (np.pi * t_i / Ts * (1 - t2**2))
            )

    return coeffs


@njit
def rcFilterTaps(t, alpha, Ts):
    r"""
    Generate Raised Cosine (RC) filter coefficients.

    Parameters
    ----------
    t : np.array
        Time values.
    alpha : float
        RC roll-off factor.
    Ts : float
        Symbol period.

    Returns
    -------
    coeffs : np.array
        RC filter coefficients.

    Notes
    -----
    The raised cosine (RC) pulse with symbol period :math:`T_s` and roll-off factor
    :math:`0 \le \alpha \le 1` is

    .. math::
        p(t) = \frac{1}{T_s}\,\mathrm{sinc}\!\left(\frac{t}{T_s}\right)
        \frac{\cos\left(\pi\alpha \frac{t}{T_s}\right)}
        {1-\left(2\alpha\frac{t}{T_s}\right)^2}, \tag{1}

    where :math:`\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)`. At the points
    :math:`t = \pm T_s/(2\alpha)`, where the denominator of Eq. (1) vanishes, the limit

    .. math::
        p\left(\pm\frac{T_s}{2\alpha}\right) = \frac{\pi}{4T_s}\,
        \mathrm{sinc}\!\left(\frac{1}{2\alpha}\right) \tag{2}

    is used. The RC pulse satisfies the Nyquist criterion for zero intersymbol
    interference, :math:`p(kT_s) = 0` for every integer :math:`k \neq 0`. Its spectrum
    is flat up to :math:`(1-\alpha)/(2T_s)` and decays with a cosine-shaped transition
    down to zero at :math:`(1+\alpha)/(2T_s)`.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.
    """
    coeffs = np.zeros(len(t), dtype=np.float64)
    π = np.pi

    for i, t_i in enumerate(t):
        t_abs = abs(t_i)
        if t_abs == Ts / (2 * alpha):
            coeffs[i] = π / (4 * Ts) * np.sinc(1 / (2 * alpha))
        else:
            coeffs[i] = (
                (1 / Ts)
                * np.sinc(t_i / Ts)
                * np.cos(π * alpha * t_i / Ts)
                / (1 - 4 * alpha**2 * t_i**2 / Ts**2)
            )

    return coeffs


def pulseShape(param):
    r"""
    Generate a pulse shaping filter.

    Parameters
    ----------
    param : optic.utils.parameters object, optional
        Parameters of the pulse shaping filter:

        - param.pulseType : Type of pulse shaping filter ('rect','nrz','rrc','rc', 'doubinary'). [default: 'rrc']
        - param.SpS : Number of samples per symbol of input signal.[default: 2]
        - param.nFilterTaps : Number of filter coefficients. [default: 256]
        - param.rollOff : Rolloff of RRC/RC filters. [default: 0.1]

    Returns
    -------
    filterCoeffs : np.array
        Array of filter coefficients (normalized).

    Notes
    -----
    The pulse :math:`p[n]` is sampled with :math:`\mathrm{SpS}` samples per symbol
    period (:math:`T_s = 1`), and the available shapes are:

    - ``'rect'``: rectangular pulse of duration :math:`T_s`.
    - ``'nrz'``: non-return-to-zero pulse, obtained by smoothing the rectangular pulse
      with a Gaussian window, which emulates the finite rise and fall times of a real
      NRZ signal.
    - ``'rrc'`` and ``'rc'``: root-raised cosine and raised cosine pulses (see
      :func:`rrcFilterTaps` and :func:`rcFilterTaps`), sampled at
      :math:`t = n/\mathrm{SpS}` over ``nFilterTaps`` samples.
    - ``'duobinary'``: sum of two sinc pulses one symbol period apart,
      :math:`p(t) = \mathrm{sinc}(t/T_s) + \mathrm{sinc}\left((t - T_s)/T_s\right)`.

    In all cases the coefficients are normalized to unit sum,

    .. math::
        \sum_{n} p[n] = 1, \tag{1}

    that is, the filter has unit gain at DC.
    """
    pulseType = getattr(param, "pulseType", "rrc")
    SpS = getattr(param, "SpS", 2)
    nFilterTaps = getattr(param, "nFilterTaps", 256)
    rollOff = getattr(param, "rollOff", 0.1)

    if pulseType == "rect":
        pulse = np.concatenate(
            (np.zeros(int(SpS / 2)), np.ones(SpS), np.zeros(int(SpS / 2)))
        )
    elif pulseType == "nrz":
        t = np.linspace(-2, 2, SpS)
        Te = 1
        pulse = np.convolve(
            np.ones(SpS),
            2 / (np.sqrt(np.pi) * Te) * np.exp(-(t**2) / Te),
            mode="full",
        )
    elif pulseType == "rrc":
        t = np.linspace(-nFilterTaps // 2, nFilterTaps // 2, nFilterTaps) * (1 / SpS)
        pulse = rrcFilterTaps(t, rollOff, 1)
    elif pulseType == "rc":
        t = np.linspace(-nFilterTaps // 2, nFilterTaps // 2, nFilterTaps) * (1 / SpS)
        pulse = rcFilterTaps(t, rollOff, 1)
    elif pulseType == "duobinary":
        t = np.linspace(
            -nFilterTaps // 2 - SpS // 2, nFilterTaps // 2 + SpS // 2, nFilterTaps
        ) * (1 / SpS)
        pulse = np.sinc(t)
        pulse += np.roll(pulse, SpS)

    pulse = pulse / np.sum(pulse)  # Normalize the filter coefficients

    return pulse


@njit(parallel=True)
def clockSamplingInterp(x, inFs, outFs, jitter=0):
    r"""
    Interpolate signal to a given sampling rate.

    Parameters
    ----------
    x : np.array
        Input signal.

    inFs : float
        Sampling frequency of the input signal.

    outFs : float
        Sampling frequency of the output signal.

    jitter : float
        Standard deviation of the time jitter (jitter rms). Default is 0.

    Returns
    -------
    y : np.array
        Resampled signal.

    Notes
    -----
    The input samples :math:`x[n] = x(nT_{in})`, with :math:`T_{in} = 1/F_{in}`, are
    resampled at the instants of the output sampling clock,

    .. math::
        t_m = m T_{out} + \epsilon_m, \qquad T_{out} = 1/F_{out}, \tag{1}

    where :math:`\epsilon_m \sim \mathcal{N}(0, \sigma_j^2)` models a random timing
    jitter with standard deviation (rms jitter) :math:`\sigma_j`. The signal value at
    :math:`t_m` is obtained by linear interpolation between the two nearest input
    samples: for :math:`nT_{in} \le t_m < (n+1)T_{in}`,

    .. math::
        y[m] = x[n] + \frac{t_m - nT_{in}}{T_{in}}\left(x[n+1] - x[n]\right). \tag{2}
    """
    nModes = x.shape[1]

    inTs = 1 / inFs
    outTs = 1 / outFs

    tin = np.arange(0, x.shape[0]) * inTs
    tout = np.arange(0, x.shape[0] * inTs, outTs)

    if jitter > 0:
        dt = np.random.normal(0, jitter, tout.shape)
        tout += dt

    y = np.zeros((len(tout), x.shape[1]), dtype=x.dtype)

    for k in prange(nModes):
        y[:, k] = np.interp(tout, tin, x[:, k])

    return y


@njit(parallel=True)
def quantizer(x, nBits=16, maxV=1, minV=-1):
    r"""
    Quantize the input signal using a uniform quantizer with the specified precision.

    Parameters
    ----------
    x : np.array
        The input signal to be quantized.
    nBits : int
        Number of bits used for quantization. The quantizer will have 2^nBits levels.
    maxV : float, optional
        Maximum value for the quantizer's full-scale range (default is 1).
    minV : float, optional
        Minimum value for the quantizer's full-scale range (default is -1).

    Returns
    -------
    np.array
        The quantized output signal with the same shape as 'x', quantized using 'nBits' levels.

    Notes
    -----
    A uniform quantizer with :math:`b` bits maps its input to one of :math:`2^b` levels
    equally spaced over the full-scale range :math:`[V_{min}, V_{max}]`,

    .. math::
        d_k = V_{min} + k\Delta, \qquad
        \Delta = \frac{V_{max} - V_{min}}{2^b - 1}, \qquad k = 0, \ldots, 2^b - 1. \tag{1}

    Each sample is replaced by the closest level,

    .. math::
        y[n] = d_{\hat{k}}, \qquad \hat{k} = \arg\min_k\, |x[n] - d_k|. \tag{2}

    For inputs within the full-scale range, the quantization error
    :math:`e[n] = y[n] - x[n]` is bounded by :math:`|e[n]| \le \Delta/2` and, for
    signals that are busy enough, it is well approximated by a uniformly distributed
    noise with power :math:`\Delta^2/12`. Inputs outside the range are mapped to the
    closest end level.
    """
    Δ = (maxV - minV) / (2**nBits - 1)

    d = np.arange(minV, maxV + Δ, Δ)
    lastLevel = len(d) - 1

    y = np.zeros(x.shape, dtype=np.float64)

    for idx in prange(len(x)):
        for indMode in range(x.shape[1]):
            xk = x[idx, indMode]

            # the closest level is at most one position away from the rounded
            # (and clipped) level index, so only these candidates are checked
            k = int(np.round(min(max((xk - minV) / Δ, 0), lastLevel)))

            closest = max(k - 1, 0)
            for level in range(closest + 1, min(k + 1, lastLevel) + 1):
                if np.abs(xk - d[level]) < np.abs(xk - d[closest]):
                    closest = level

            y[idx, indMode] = d[closest]

    return y


def lowPassFIR(fc, fs, N, typeF="rect"):
    r"""
    Calculate FIR coefficients of a lowpass filter.

    Parameters
    ----------
    fc : float
        Cutoff frequency.
    fs : float
        Sampling frequency.
    N : int
        Number of filter coefficients.
    typeF : string, optional
        Type of response ('rect', 'gauss'). The default is "rect".

    Returns
    -------
    h : np.array
        Filter coefficients.

    Notes
    -----
    With the normalized cutoff frequency :math:`f_u = f_c/f_s` and the filter delay
    :math:`d = (N-1)/2`, two responses are available:

    - ``'rect'``: truncated impulse response of the ideal lowpass filter, whose
      frequency response is rectangular with cutoff :math:`f_c`,

      .. math::
          h[n] = 2f_u\,\mathrm{sinc}\left(2f_u(n-d)\right), \tag{1}

      where :math:`\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)`.

    - ``'gauss'``: Gaussian filter, with frequency response
      :math:`H(f) = \exp\left[-\frac{\ln 2}{2}\left(f/f_c\right)^2\right]`, so that
      :math:`|H(f_c)|^2 = 1/2` (3-dB cutoff at :math:`f_c`), and impulse response

      .. math::
          h[n] = \sqrt{\frac{2\pi}{\ln 2}}\, f_u
          \exp\left[-\frac{2}{\ln 2}\left(\pi f_u (n-d)\right)^2\right]. \tag{2}

    In both cases the coefficients are then normalized to unit sum (unit gain at DC).

    References
    ----------
    [1] P. S. R. Diniz, E. A. B. da Silva, e S. L. Netto, Digital Signal Processing: System Analysis and Design. Cambridge University Press, 2010.

    """
    fu = fc / fs
    d = (N - 1) / 2
    n = np.arange(0, N)

    # calculate filter coefficients
    if typeF == "rect":
        h = (2 * fu) * np.sinc(2 * fu * (n - d))
    elif typeF == "gauss":
        h = (
            np.sqrt(2 * np.pi / np.log(2))
            * fu
            * np.exp(-(2 / np.log(2)) * (np.pi * fu * (n - d)) ** 2)
        )
    h = h / np.sum(h)  # Normalize the filter coefficients

    return h


def upsample(x, factor):
    r"""
    Upsample a signal by inserting zeros between samples.

    Parameters
    ----------
    x : np.array
        Input signal to upsample.
    factor : int
        Upsampling factor. The signal will be upsampled by inserting
        `factor - 1` zeros between each original sample.

    Returns
    -------
    xUp : np.array
        The upsampled signal with zeros inserted between samples.

    Notes
    -----
    This function inserts zeros between the samples of the input signal to
    increase its sampling rate. The upsampling factor determines how many
    zeros are inserted between each original sample.

    If the input signal is a 2D array, the upsampling is performed
    column-wise.

    Formally, upsampling by an integer factor :math:`L` is defined as

    .. math::
        :nowrap:

        \begin{equation}
            x_{\uparrow}[n] =
            \begin{cases}
                x[n/L], & n = 0, \pm L, \pm 2L, \ldots \\
                0, & \text{otherwise.}
            \end{cases} \tag{1}
        \end{equation}

    In the frequency domain, :math:`X_{\uparrow}(e^{j\omega}) = X(e^{j\omega L})`: the
    spectrum is compressed by :math:`L` and :math:`L-1` spectral images appear in
    :math:`[-\pi, \pi)`. These images are removed by a subsequent interpolation (e.g.
    pulse shaping) filter.

    References
    ----------
    [1] P. S. R. Diniz, E. A. B. da Silva, e S. L. Netto, Digital Signal Processing: System Analysis and Design. Cambridge University Press, 2010.
    """
    try:
        xUp = np.zeros((factor * x.shape[0], x.shape[1]), dtype=x.dtype)
        xUp[0::factor, :] = x
    except IndexError:
        xUp = np.zeros(factor * x.shape[0], dtype=x.dtype)
        xUp[0::factor] = x

    return xUp


def decimate(sigIn, param):
    r"""
    Decimate signal.

    Parameters
    ----------
    sigIn : np.array
        Input signal.
    param : optic.utils.parameters object, optional
        Parameters of the decimation process.

        - param.SpSin : samples per symbol of the input signal.
        - param.SpSout : samples per symbol of the output signal.

    Returns
    -------
    sigOut : np.array
        Decimated signal.

    Notes
    -----
    The signal has :math:`\mathrm{SpS}_{in}` samples per symbol and is reduced to
    :math:`\mathrm{SpS}_{out}` samples per symbol by keeping one out of every
    :math:`D = \mathrm{SpS}_{in}/\mathrm{SpS}_{out}` samples. The sampling phase
    :math:`n_0` is chosen as the position within the symbol period where the signal
    variance is maximum, which, for a Nyquist-shaped signal, corresponds to the center
    of the eye diagram:

    .. math::
        n_0 = \arg\max_{k \in \{0, \ldots, \mathrm{SpS}_{in}-1\}}
        \mathrm{Var}\left\{x[m\,\mathrm{SpS}_{in} + k]\right\}_m, \tag{1}

    .. math::
        y[m] = x[n_0 + mD]. \tag{2}

    References
    ----------
    [1] P. S. R. Diniz, E. A. B. da Silva, e S. L. Netto, Digital Signal Processing: System Analysis and Design. Cambridge University Press, 2010.

    """
    sigIn = sigIn.copy()
    try:
        sigIn.shape[1]
        input1D = False
    except IndexError:
        input1D = True
        # If sigIn is a 1D array, reshape it to a 2D array
        sigIn = sigIn.reshape(len(sigIn), 1)

    decFactor = int(param.SpSin / param.SpSout)

    # simple timing recovery
    sampDelay = np.zeros(sigIn.shape[1])

    # finds best sampling instant
    # (maximum variance sampling time)
    for k in range(sigIn.shape[1]):
        a = sigIn[:, k].reshape(sigIn.shape[0], 1)
        a = np.reshape(sigIn[:, k], (sigIn.shape[0], 1))
        varVector = np.var(a.reshape(-1, param.SpSin), axis=0)
        sampDelay[k] = np.where(varVector == np.amax(varVector))[0][0]
    # downsampling
    sigOut = sigIn[::decFactor, :].copy()

    for k in range(sigIn.shape[1]):
        sigIn[:, k] = np.roll(sigIn[:, k], -int(sampDelay[k]))
        sigOut[:, k] = sigIn[0::decFactor, k]

    if input1D:
        # If the output is 1D, return it as a 1D array
        sigOut = sigOut.flatten()

    return sigOut


def resample(sigIn, param):
    """
    Resample signal to a desired sampling rate.

    Parameters
    ----------
    sigIn : np.array
        Input signal.
    param : optic.utils.parameters object, optional
        Parameters of the resampling process.

            - param.inFs : sampling rate of the input signal [default: 2].
            - param.outFs : sampling rate of the output signal [default: 2].
            - param.N : order of anti-aliasing filter [default: 501].

    Returns
    -------
    sigOut : np.array
        Resampled signal.

    Notes
    -----
    The signal is converted from the sampling rate :math:`F_{in}` to :math:`F_{out}` by
    linear interpolation (see :func:`clockSamplingInterp`). To avoid aliasing when
    :math:`F_{out} < F_{in}`, the signal is first filtered by a lowpass filter with
    cutoff :math:`F_{out}/2`. When :math:`F_{out} > F_{in}`, the interpolated signal is
    lowpass filtered with cutoff :math:`F_{in}/2`, which removes the spectral images
    created by the interpolation. Both filters are windowed-sinc filters with
    ``N`` coefficients (see :func:`lowPassFIR`).

    References
    ----------
    [1] P. S. R. Diniz, E. A. B. da Silva, e S. L. Netto, Digital Signal Processing: System Analysis and Design. Cambridge University Press, 2010.

    """
    # check input parameters
    N = getattr(param, "N", 501)
    inFs = getattr(param, "inFs", 2)
    outFs = getattr(param, "outFs", 2)

    try:
        sigIn.shape[1]
        input1D = False
    except IndexError:
        input1D = True
        # If sigIn is a 1D array, reshape it to a 2D array
        sigIn = sigIn.reshape(len(sigIn), 1)

    # Anti-aliasing filters:
    if outFs < inFs:
        N_ = min(sigIn.shape[0], N)
        hi = lowPassFIR(outFs / 2, inFs, N_, typeF="rect")
        sigIn = firFilter(hi, sigIn)

    sigOut = clockSamplingInterp(sigIn, inFs, outFs)

    if outFs > inFs:
        N_ = min(sigOut.shape[0], N)
        ho = lowPassFIR(inFs / 2, outFs, N_, typeF="rect")
        sigOut = firFilter(ho, sigOut)

    if input1D:
        # If the output is 1D, return it as a 1D array
        sigOut = sigOut.flatten()

    return sigOut


def symbolSync(rx, tx, SpS, mode="amp"):
    """
    Symbol synchronizer.

    Parameters
    ----------
    rx : np.array
        Received symbol sequence.
    tx : np.array
        Transmitted symbol sequence.
    SpS : int
        Samples per symbol of the received signal.
    mode : string, optional
        Synchronization mode: "amp" (amplitude) or "real" (real part). The default is "amp".

    Returns
    -------
    tx : np.array
        Transmitted sequence synchronized to rx.

    """
    try:
        rx.shape[1]
        input1D = False
    except IndexError:
        input1D = True
        # If rx is a 1D array, reshape it to a 2D array
        rx = rx.reshape(len(rx), 1)

    try:
        tx.shape[1]
    except IndexError:
        # If tx is a 1D array, reshape it to a 2D array
        tx = tx.reshape(len(tx), 1)

    nModes = rx.shape[1]

    if SpS > 1:
        # decimate received signal
        paramDec = parameters()
        paramDec.SpSin = SpS
        paramDec.SpSout = 1

        rx = decimate(rx, paramDec)

    # calculate time delay
    delay = np.zeros(nModes)

    corrMatrix = np.zeros((nModes, nModes))
    rot = np.ones((nModes, nModes), dtype=np.complex64)

    if mode == "amp":
        for n in range(nModes):
            for m in range(nModes):
                abs_tx = np.abs(tx[:, m])
                abs_tx -= np.mean(abs_tx)

                abs_rx = np.abs(rx[:, n])
                abs_rx -= np.mean(abs_rx)
                corrMatrix[m, n] = np.max(np.abs(signal.correlate(abs_tx, abs_rx)))

        swap = np.argmax(corrMatrix, axis=0)

        tx = tx[:, swap]

        for k in range(nModes):
            abs_tx = np.abs(tx[:, k])
            abs_tx -= np.mean(abs_tx)
            abs_rx = np.abs(rx[:, k])
            abs_rx -= np.mean(abs_rx)
            delay[k] = finddelay(abs_tx, abs_rx)

    elif mode == "real":
        for n in range(nModes):
            for m in range(nModes):
                crr = signal.correlate(np.real(tx[:, m]), np.real(rx[:, n]))
                cir = signal.correlate(np.imag(tx[:, m]), np.real(rx[:, n]))

                crr_peak = crr[np.argmax(np.abs(crr))]
                cir_peak = cir[np.argmax(np.abs(cir))]

                crr_peak_abs = np.abs(crr_peak)
                cir_peak_abs = np.abs(cir_peak)

                corrMatrix[m, n] = np.max([crr_peak_abs, cir_peak_abs])

                # handle pi/2 rotations
                if crr_peak_abs > cir_peak_abs:
                    if crr_peak > 0:
                        rot[m, n] = 1
                    else:
                        rot[m, n] = -1
                else:
                    if cir_peak > 0:
                        rot[m, n] = -1j
                    else:
                        rot[m, n] = 1j

        swap = np.argmax(corrMatrix, axis=0)
        tx = tx[:, swap]

        for k in range(nModes):
            # apply rotation
            tx[:, k] = rot[k, swap[k]] * tx[:, k]

            # calculate delay
            delay[k] = finddelay(np.real(tx[:, k]), np.real(rx[:, k]))

            # check if conjugation is needed
            cii = signal.correlate(np.imag(tx[:, k]), np.imag(rx[:, k]))
            cii_peak = cii[np.argmax(np.abs(cii))]

            if cii_peak < 0:
                tx[:, k] = tx[:, k].conj()

    # compensate time delay
    for k in range(nModes):
        tx[:, k] = np.roll(tx[:, k], -int(delay[k]))

    if input1D:
        # If the output is 1D, return it as a 1D array
        tx = tx.flatten()

    return tx


def finddelay(x, y):
    r"""
    Find delay between x and y.

    Parameters
    ----------
    x : np.array
        Signal 1.
    y : np.array
        Signal 2.

    Returns
    -------
    d : int
        Delay between x and y, in samples.

    Notes
    -----
    The delay is estimated from the peak of the magnitude of the cross-correlation
    between the two sequences,

    .. math::
        R_{xy}[\tau] = \sum_{n} x[n + \tau]\, y^*[n], \qquad
        \hat{\tau} = \arg\max_{\tau}\, |R_{xy}[\tau]|, \tag{1}

    so that :math:`y[n] \approx x[n + \hat{\tau}]`. Using the magnitude of the
    correlation makes the estimate insensitive to a constant phase rotation between
    the sequences.
    """
    xcorr = np.abs(signal.correlate(x, y))
    delay = np.argmax(xcorr) - x.shape[0] + 1

    return delay


@njit
def pnorm(x):
    r"""
    Normalize the average power of each componennt of x.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    np.array
        Signal x with each component normalized in power.

    Notes
    -----
    The signal is scaled to unit average power,

    .. math::
        y[n] = \frac{x[n]}{\sqrt{P_x}}, \qquad
        P_x = \frac{1}{N}\sum_{n} |x[n]|^2, \tag{1}

    where the average power :math:`P_x` is computed over all the elements of
    :math:`x`.
    """
    return x / np.sqrt(np.mean(x * np.conj(x)).real)


@njit
def anorm(x):
    r"""
    Normalize the amplitude of each componennt of x.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    np.array
        Signal x with each component normalized in amplitude.

    Notes
    -----
    The signal is scaled so that its peak magnitude is equal to one,

    .. math::
        y[n] = \frac{x[n]}{\max_{m} |x[m]|}, \tag{1}

    where the maximum is taken over all the elements of :math:`x`.
    """
    return x / np.max(np.abs(x))


@njit
def gaussianComplexNoise(shapeOut, σ2=1.0, seed=None):
    r"""
    Generate complex circular Gaussian noise.

    Parameters
    ----------
    shapeOut : tuple of int
        Shape of np.array to be generated.
    σ2 : float, optional
        Variance of the noise (default is 1).
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    noise : np.array
        Generated complex circular Gaussian noise.

    Notes
    -----
    The samples are drawn from a zero-mean circularly-symmetric complex Gaussian
    distribution, :math:`n = n_I + j n_Q`, where the in-phase and quadrature
    components are independent and identically distributed,
    :math:`n_I, n_Q \sim \mathcal{N}(0, \sigma^2/2)`. Hence
    :math:`\mathbb{E}\left[|n|^2\right] = \sigma^2` and the probability density
    function is

    .. math::
        p(n) = \frac{1}{\pi\sigma^2}\exp\left(-\frac{|n|^2}{\sigma^2}\right). \tag{1}

    This is the standard model for additive white Gaussian noise (AWGN) in complex
    baseband.
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.normal(0, np.sqrt(σ2 / 2), shapeOut) + 1j * np.random.normal(
        0, np.sqrt(σ2 / 2), shapeOut
    )


@njit
def gaussianNoise(shapeOut, σ2=1.0, seed=None):
    r"""
    Generate Gaussian noise.

    Parameters
    ----------
    shapeOut : tuple of int
        Shape of np.array to be generated.
    σ2 : float, optional
        Variance of the noise (default is 1).
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    noise : np.array
        Generated Gaussian noise.

    Notes
    -----
    The samples are drawn from a zero-mean real Gaussian distribution with variance
    :math:`\sigma^2`,

    .. math::
        p(n) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{n^2}{2\sigma^2}\right). \tag{1}
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.normal(0, np.sqrt(σ2), shapeOut)


@njit
def phaseNoise(lw, Nsamples, Ts, seed=None):
    r"""
    Generate realization of a random-walk phase-noise process.

    Parameters
    ----------
    lw : scalar
        laser linewidth.
    Nsamples : scalar
        number of samples to be draw.
    Ts : scalar
        sampling period.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    phi : np.array
        realization of the phase noise process.

    Notes
    -----
    The phase noise of a laser with linewidth :math:`\Delta\nu` is modeled as a Wiener
    process (random walk), whose increments over a sampling period :math:`T_s` are
    independent zero-mean Gaussian random variables:

    .. math::
        \phi[k+1] = \phi[k] + \Delta_k, \qquad
        \Delta_k \sim \mathcal{N}\left(0,\, \sigma^2\right), \qquad
        \sigma^2 = 2\pi\Delta\nu T_s, \tag{1}

    with :math:`\phi[0] = 0`. The variance of the phase difference accumulated over an
    interval :math:`\tau` grows linearly with it, :math:`2\pi\Delta\nu|\tau|`, and the
    corresponding optical field :math:`e^{j\phi(t)}` has a Lorentzian power spectral
    density with full width at half maximum equal to :math:`\Delta\nu`.

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    """
    if seed is not None:
        np.random.seed(seed)

    σ2 = 2 * np.pi * lw * Ts
    phi = np.zeros(Nsamples)

    for ind in range(Nsamples - 1):
        phi[ind + 1] = phi[ind] + np.random.normal(0, np.sqrt(σ2))

    return phi


def movingAverage(x, N):
    r"""
    Calculate the sliding window moving average of a 2D NumPy array along each column.

    Parameters
    ----------
    x : np.array
        Input 2D array with shape (M, N), where M is the number of samples and N is the number of columns.
    N : int
        Size of the sliding window.

    Returns
    -------
    np.array
        2D array containing the sliding window moving averages along each column.

    Notes
    -----
    The function pads the signal with zeros at both ends to compensate for the lag between the output
    of the moving average and the original signal.

    The moving average over a window of :math:`N` samples centered at each output
    sample is the FIR filter with :math:`N` equal coefficients :math:`1/N`,

    .. math::
        y[n] = \frac{1}{N}\sum_{k=-\lfloor N/2 \rfloor}^{N - 1 - \lfloor N/2 \rfloor}
        x[n+k], \tag{1}

    with :math:`x[n] = 0` outside the signal (zero padding at the edges).
    """
    try:
        x.shape[1]
        input1D = False
    except IndexError:
        x = x.reshape(len(x), 1)
        input1D = True

    nCol = x.shape[1]
    y = np.zeros(x.shape, dtype=x.dtype)

    startInd = N // 2

    endInd = -N // 2 + 1 if N % 2 else -N // 2
    for indCol in range(nCol):
        # Pad the signal with zeros at both ends
        padded_x = np.pad(x[:, indCol], (N // 2, N // 2), mode="constant")

        # Calculate moving average using convolution
        h = np.ones(N) / N
        # ma = np.convolve(padded_x, h, "same")
        ma = signal.fftconvolve(padded_x, h, "same")
        y[:, indCol] = ma[startInd:endInd]

    if input1D:
        y = y.flatten()

    return y


def delaySignal(sig, delay, Fs=1, NFFT=1024):
    r"""
    Apply a time delay to a signal sampled at fs samples per second using FFT/IFFT algorithms.

    Parameters
    ----------
    sig : np.array
        The input signal.
    delay : float
        The time delay to apply to the signal (in seconds).
    Fs : float
        Sampling frequency of the signal (in samples per second). Default is 1.
    NFFT : int, optional
        FFT size to be used. Must be greater than the length of the filter.
        If None, it will be set to the next power of 2 greater than or equal
        to the length of the (zero-padded) signal. Default is 1024.

    Returns
    -------
    np.array
        The delayed signal, with the same length as `sig`. If `delay` is zero,
        a copy of `sig` is returned.

    Notes
    -----
    A time delay :math:`\tau` corresponds to a linear phase in the frequency domain,

    .. math::
        y(t) = x(t - \tau) \quad \Longleftrightarrow \quad
        Y(f) = X(f)\, e^{-j2\pi f\tau}, \tag{1}

    which holds for any real :math:`\tau`, including fractions of the sampling period.
    The frequency response in Eq. (1) is applied by blockwise FFT convolution
    (overlap-and-save, see :func:`blockwiseFFTConv`), after zero padding the signal to
    avoid the circular wrap-around of the delayed samples.
    """
    if delay == 0:
        # nothing to delay: return a copy with the same output dtype
        return sig.astype(np.result_type(sig.dtype, np.float64))

    # Calculate the length of the signal
    N = len(sig)

    # Calculate the length of zero padding needed
    padLen = int(np.ceil(np.abs(delay * Fs)))

    # Zero-pad the signal to avoid circular shift (plus one sample to hold the
    # extra one-sample delay of the even-length frequency-domain filter)
    sigPad = np.pad(sig, (0, padLen + 1), mode="constant")

    if NFFT is None:
        NFFT = 2 ** int(np.ceil(np.log2(N + padLen + 1)))

    # Compute the frequency vector
    freq = fftfreq(NFFT // 2, d=1 / Fs)

    # Apply the phase shift corresponding to the time delay
    H = np.exp(-1j * 2 * np.pi * freq * delay)
    delayedSig = blockwiseFFTConv(sigPad, H, NFFT=NFFT, freqDomainFilter=True)

    # discard the extra one-sample filter delay
    return delayedSig[1 : N + 1]


def iqMixing(sig, param):
    r"""
    Add IQ mixing to a signal.

    Parameters
    ----------
    sig : np.array
        Input signal.
    param : optic.utils.parameters object
        Parameters of IQ mixing.

        param.ampImb : Amplitude imbalance parameter in dB.[default: 0 dB]
        param.phaseImb : Phase imbalance parameter (in radians).[default: 0 rad]
        param.timeSkew : delay between I and Q components. [default: 0 s]
        param.Fs : simulation sampling frequency. [default: None]

    Returns
    -------
    np.array
        IQ-mixed signal.

    Notes
    -----
    Imperfections of the in-phase (I) and quadrature (Q) branches of a receiver
    front-end are modeled as follows. For an input :math:`s = I + jQ`, an amplitude
    imbalance :math:`\epsilon` and a phase imbalance :math:`\phi` between the
    branches result in

    .. math::
        y_I = (1-\epsilon)\left[I\cos\frac{\phi}{2} - Q\sin\frac{\phi}{2}\right], \qquad
        y_Q = (1+\epsilon)\left[Q\cos\frac{\phi}{2} - I\sin\frac{\phi}{2}\right], \tag{1}

    where :math:`\epsilon = 10^{A_{dB}/20} - 1` is obtained from the amplitude
    imbalance in dB. Equivalently, in complex notation, the IQ imbalance adds an image
    of the complex conjugate of the signal,

    .. math::
        y = k_1 s + k_2 s^*, \tag{2}

    with :math:`k_1 = \left[(1-\epsilon)e^{j\phi/2} + (1+\epsilon)e^{-j\phi/2}\right]/2`
    and :math:`k_2 = \left[(1-\epsilon)e^{-j\phi/2} - (1+\epsilon)e^{j\phi/2}\right]/2`.
    Finally, a time skew :math:`\tau` between the branches is applied by advancing the
    I component and delaying the Q component by :math:`\tau/2`.
    """
    # check input parameters
    ampImb = getattr(param, "ampImb", 0)
    phaseImb = getattr(param, "phaseImb", 0)
    timeSkew = getattr(param, "timeSkew", 0)
    Fs = getattr(param, "Fs", None)

    if Fs is None:
        logg.error("Sampling frequency not provided.")

    # IQ-imbalance
    ampImb = 10 ** (ampImb / 20) - 1  # convert from dB to linear scale
    k1 = (1 - ampImb) * np.exp(1j * phaseImb / 2) / 2 + (1 + ampImb) * np.exp(
        -1j * phaseImb / 2
    ) / 2
    k2 = (1 - ampImb) * np.exp(-1j * phaseImb / 2) / 2 - (1 + ampImb) * np.exp(
        1j * phaseImb / 2
    ) / 2
    sig_ = k1 * sig + k2 * np.conj(sig)

    # IQ-skew
    delay = timeSkew / 2
    sI = delaySignal(np.real(sig_), -delay, Fs).real
    sQ = delaySignal(np.imag(sig_), delay, Fs).real

    return sI + 1j * sQ


def blockwiseFFTConv(x, h, NFFT=None, freqDomainFilter=False):
    r"""
    Blockwise convolution in the frequency domain using the overlap-and-save FFT method.

    Parameters
    ----------
    x : np.array
        Input signal.
    h : np.array
        Filter impulse response.
    NFFT : int, optional
        FFT size to be used. Must be greater than the length of the filter.
        If None, it will be set to the next power of 2 greater than or equal
        to the length of the filter. Default is None.
    freqDomainFilter : bool, optional
        If True, `h` is assumed to be the frequency response of the filter.
        If False, the FFT of `h` will be computed. Default is False.

    Returns
    -------
    y : np.array
        The filtered output signal.

    Raises
    ------
    ValueError
        If NFFT is not greater than the length of the filter `h`.

    Notes
    -----
    Long convolutions are computed in the frequency domain with the overlap-and-save
    method. The input is split into overlapping blocks of :math:`N_{FFT}` samples, each
    one sharing its first :math:`K-1` samples with the previous block, where :math:`K`
    is the length of the filter impulse response. For each block :math:`b`,

    .. math::
        y_b = \mathrm{IDFT}\left\{\mathrm{DFT}\{x_b\} \cdot H\right\}, \tag{1}

    where :math:`H` is the :math:`N_{FFT}`-point DFT of the zero-padded impulse
    response. The first :math:`K-1` samples of :math:`y_b` are corrupted by the
    circular wrap-around of the DFT and are discarded, while the remaining
    :math:`N_{FFT} - K + 1` samples are equal to the linear convolution and are
    concatenated to form the output. The filter delay :math:`\lfloor (K-1)/2 \rfloor`
    is compensated. The computational cost grows as :math:`\mathcal{O}(N\log N_{FFT})`
    instead of :math:`\mathcal{O}(NK)` for the direct convolution.
    """
    sigLen = len(x)  # length of the input signal
    K = len(h)  # length of the filter impulse response
    D = (K - 1) // 2  # filter delay

    if NFFT is None:
        NFFT = 2 ** int(np.ceil(np.log2(np.max([sigLen, K]))))

    if NFFT >= K:
        d = NFFT - K + 1  # block length required
    else:
        logg.error("FFT size is smaller than filter length")

    if freqDomainFilter:
        # Assumes h is frequency response centered at DC
        h = np.pad(fftshift(ifft(h)), (0, NFFT - K), mode="constant")
    else:
        h = np.pad(h, (0, NFFT - K), mode="constant")

    H = fft(h)  # frequency response

    discard = K - 1  # number of samples to be discarded after IFFT (overlap samples)
    numBlocks = int(
        np.ceil((sigLen + K - 1) / d)
    )  # total number of FFT blocks to be processed

    padLen = numBlocks * d + discard - sigLen

    # pad signal with padLen zeros + D zeros (to compensate for filter delay)
    x = np.pad(x, (discard, padLen + D), mode="constant")

    # pre-allocate output
    y = np.zeros(numBlocks * d, dtype="complex")

    for blk in range(numBlocks):
        # extract block and compute FFT
        X = fft(x[blk * d : blk * d + NFFT])
        # frequency domain multiplication and IFFT
        y_blk = ifft(X * H)
        # save valid part of the block
        y[blk * d : (blk + 1) * d] = y_blk[discard:]

    if np.any(np.iscomplex(x)):
        return y[D : D + sigLen]
    else:
        return y[D : D + sigLen].real


@njit
def freqShift(x, deltaF, Fs):
    r"""
    Frequency shift of a signal.

    Parameters
    ----------
    x : np.array
        Input signal.
    deltaF : float
        Frequency shift (Hz).
    Fs : float
        Sampling frequency (Hz).

    Returns
    -------
    y : np.array
        Frequency shifted signal.

    Notes
    -----
    A frequency shift :math:`\Delta f` is the multiplication by a complex exponential
    (modulation theorem),

    .. math::
        y[n] = x[n]\, e^{j2\pi\Delta f\, n/F_s}
        \quad \Longleftrightarrow \quad
        Y(f) = X(f - \Delta f), \tag{1}

    where :math:`F_s` is the sampling frequency.
    """
    t = np.arange(len(x)) * (1 / Fs)
    y = x * np.exp(1j * 2 * np.pi * deltaF * t)

    return y


@njit
def calcMZM(sigIn, Vpi, u, Vb, ER):
    r"""
    Fast function to calculate the Mach-Zehnder modulator (MZM) model.

    Parameters
    ----------
    sigIn : np.array or float
        Complex-valued optical input field.
    Vpi : float
        Half-wave voltage of the MZM.
    u : float
        RF voltage applied to the MZM.
    Vb : float
        DC bias voltage.
    ER : float
        Extinction ratio of the MZM (in dB).

    Returns
    -------
    np.array or float
        Complex-valued optical output field after modulation.

    Notes
    -----
    A Mach-Zehnder modulator (MZM) splits the input field between two arms, applies
    opposite phase shifts :math:`\pm\theta` to them (push-pull operation), and
    recombines the two fields. With the drive voltage :math:`u(t)` and the bias
    :math:`V_b`, the phase shift in each arm is

    .. math::
        \theta(t) = \frac{\pi}{2}\frac{u(t) + V_b}{V_\pi}. \tag{1}

    A finite extinction ratio :math:`\varepsilon = 10^{ER/10}` is modeled by an
    imbalance between the fields of the arms,

    .. math::
        E_{out}(t) = \frac{E_{in}(t)}{2}\left[\sqrt{1+\gamma}\,e^{j\theta(t)}
        + \sqrt{1-\gamma}\,e^{-j\theta(t)}\right], \qquad
        \gamma = \frac{2\sqrt{\varepsilon}}{\varepsilon + 1}, \tag{2}

    which can be written as

    .. math::
        E_{out}(t) = E_{in}(t)\left[c_I\cos\theta(t) + jc_Q\sin\theta(t)\right], \qquad
        c_{I,Q} = \frac{\sqrt{1+\gamma} \pm \sqrt{1-\gamma}}{2}. \tag{3}

    The ratio between the maximum and the minimum output powers is
    :math:`c_I^2/c_Q^2 = \varepsilon`. For an infinite extinction ratio,
    :math:`c_I = 1` and :math:`c_Q = 0`, and Eq. (3) reduces to the transfer function
    of the ideal MZM, :math:`E_{out} = E_{in}\cos\theta`.

    References
    ----------
    [1] Y. Yamaguchi, et al, "Precise Optical Modulation Using Extinction-Ratio and Chirp Tunable Single-Drive Mach–Zehnder Modulator," Journal of Lightwave Technology, vol. 35, no. 21, pp. 4781-4788, 1 Nov.1, 2017,

    [2] Seimetz, M., High-Order Modulation for Optical Fiber Transmission. Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.
    """
    # convert extinction ratio from dB to linear scale
    erLin = 10 ** (ER / 10)
    gamma = 2 * np.sqrt(erLin) / (erLin + 1)

    # The two arms apply opposite phase shifts ±θ, with amplitudes sqrt(1 ± gamma):
    # [sqrt(1 + gamma) * exp(jθ) + sqrt(1 - gamma) * exp(-jθ)] / 2 = cI * cos(θ) + j * cQ * sin(θ)
    cI = (np.sqrt(1 + gamma) + np.sqrt(1 - gamma)) / 2
    cQ = (np.sqrt(1 + gamma) - np.sqrt(1 - gamma)) / 2
    θ = (u + Vb) / (2 * Vpi) * np.pi

    return sigIn * (cI * np.cos(θ) + 1j * cQ * np.sin(θ))


@njit
def calcPM(sigIn, Vpi, u):
    r"""
    Fast function to calculate the phase modulator (PM) model.

    Parameters
    ----------
    sigIn : np.array or float
        Complex-valued optical input field.
    Vpi : float
        Half-wave voltage of the PM.
    u : float
        Driving voltage applied to the PM.

    Returns
    -------
    np.array or float
        Complex-valued optical output field after modulation.

    Notes
    -----
    An optical phase modulator (PM) driven by the voltage :math:`u(t)` imposes a phase
    shift proportional to it,

    .. math::
        E_{out}(t) = E_{in}(t)\exp\left[j\pi\frac{u(t)}{V_\pi}\right], \tag{1}

    where :math:`V_\pi` is the voltage that produces a phase shift of :math:`\pi` rad.

    References
    ----------
    [1] Seimetz, M., High-Order Modulation for Optical Fiber Transmission. Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    """
    φ = (u / Vpi) * np.pi

    return sigIn * (np.cos(φ) + 1j * np.sin(φ))  # sigIn * exp(jφ)


@njit(fastmath=True, cache=True)
def levinson(r, nTaps):
    r"""
    Levinson-Durbin algorithm

    Parameters
    ----------
    r : array-like
        Autocorrelation coefficients of the signal, where r[0] is the zero-lag autocorrelation and r[k] is the autocorrelation at lag k.
    nTaps : int
        The order of the whitening filter (number of coefficients to estimate).

    Returns
    -------
    a : np.np.array
        The coefficients of the whitening filter of length nTaps, where a[0] is
        the leading coefficient (usually 1) and a[1], a[2], ..., a[nTaps-1] are the estimated filter coefficients.

    Notes
    -----
    The Levinson-Durbin algorithm is an efficient method for solving the Toeplitz system of equations that arises in linear prediction and filter design.
    The resulting coefficients can be used to design a whitening filter that decorrelates the input signal.

    In linear prediction of order :math:`p = n_{Taps} - 1`, the prediction error
    filter :math:`A(z) = \sum_{k=0}^{p} a_k z^{-k}`, with :math:`a_0 = 1`, minimizes the
    power of :math:`e[n] = \sum_{k=0}^{p} a_k x[n-k]`. Its coefficients satisfy the
    Yule-Walker (normal) equations, whose matrix is Toeplitz and Hermitian,

    .. math::
        \sum_{k=1}^{p} a_k\, r[i-k] = -r[i], \qquad i = 1, \ldots, p, \tag{1}

    where :math:`r[k]` is the autocorrelation of :math:`x[n]`. The Levinson-Durbin
    recursion solves Eq. (1) in :math:`\mathcal{O}(p^2)` operations, increasing the
    order one step at a time. Starting from :math:`E_0 = r[0]`, for
    :math:`i = 1, \ldots, p`:

    .. math::
        k_i = -\frac{r[i] + \sum_{j=1}^{i-1} a_j^{(i-1)}\, r[i-j]}{E_{i-1}}, \tag{2}

    .. math::
        a_j^{(i)} = a_j^{(i-1)} + k_i\, a_{i-j}^{(i-1)*}, \quad j = 1, \ldots, i-1,
        \qquad a_i^{(i)} = k_i, \tag{3}

    .. math::
        E_i = \left(1 - |k_i|^2\right) E_{i-1}, \tag{4}

    where :math:`k_i` are the reflection coefficients and :math:`E_i` is the
    prediction error power of order :math:`i`.

    References
    ----------
    [1] Levinson, N., The Wiener RMS error criterion in filter design. Journal of Mathematics and Physics, 25(1-4), 261-278, 1947.

    [2] Durbin, J., The fitting of time-series models. Review of the International Statistical Institute, 28(3), 233-244, 1960.
    """
    a = np.zeros(nTaps, dtype=r.dtype)
    e = r[0]
    a[0] = 1.0

    for i in range(1, nTaps):
        acc = 0
        for j in range(1, i):
            acc += a[j] * r[i - j]

        k = -(r[i] + acc) / e

        a_new = a.copy()
        for j in range(1, i):
            a_new[j] += k * np.conj(a[i - j])

        a_new[i] = k
        a = a_new
        e *= 1 - np.abs(k) ** 2

    return a


@njit(fastmath=True, cache=True)
def autocorr(x, nTaps):
    r"""
    Estimate the autocorrelation coefficients of a signal x up to lag nTaps-1.

    Parameters
    ----------
    x : array-like
        The input signal for which to estimate the autocorrelation coefficients.
    nTaps : int
        The number of autocorrelation coefficients to estimate (lags from 0 to nTaps-1).
    Returns
    -------
    r : np.array
        An array of length nTaps containing the estimated autocorrelation coefficients, where r[k] is the autocorrelation at lag k.

    Notes
    -----
    The autocorrelation coefficients are estimated using the unbiased estimator, which normalizes the sum of products by the number of terms that contribute to each lag. This provides a more accurate estimate of the autocorrelation, especially for larger lags.

    Explicitly, the estimate at lag :math:`k` is

    .. math::
        \hat{r}[k] = \frac{1}{N-k}\sum_{n=k}^{N-1} x[n]\, x^*[n-k],
        \qquad k = 0, \ldots, n_{Taps}-1, \tag{1}

    where :math:`N` is the length of the signal.

    References
    ----------

    [1] Gallager, R. G., Introduction to Random Signals and Applied Kalman Filtering. John Wiley & Sons, 2010.
    """
    N = len(x)
    r = np.zeros(nTaps)

    for k in range(nTaps):
        for n in range(k, N):
            r[k] += x[n] * np.conj(x[n - k])

        r[k] /= N - k

    return r


@njit(cache=True)
def estimateWhiteningFilter(x, nTaps):
    r"""
    Estimate the coefficients of a whitening filter of order nTaps using the Levinson-Durbin algorithm.

    Parameters
    ----------
    x : array-like
        The input signal from which to estimate the autocorrelation.
    nTaps : int
        The order of the whitening filter (number of coefficients).
    Returns
    -------
    w : np.array
        The coefficients of the whitening filter of length nTaps, where w[0] is the leading coefficient (usually 1).

    Notes
    -----
    The whitening filter is the prediction error filter of the signal: the
    autocorrelation of :math:`x[n]` is estimated with :func:`autocorr`, and the
    coefficients :math:`a_k` are obtained by solving the Yule-Walker equations with
    the Levinson-Durbin recursion (:func:`levinson`). The output of the filter,

    .. math::
        e[n] = \sum_{k=0}^{n_{Taps}-1} a_k\, x[n-k], \qquad a_0 = 1, \tag{1}

    is the prediction error, which is approximately white when the filter order is
    large enough to capture the correlation of :math:`x[n]`.

    References
    ----------
    [1] Levinson, N., The Wiener RMS error criterion in filter design. Journal of Mathematics and Physics, 25(1-4), 261-278, 1947.

    [2] Durbin, J., The fitting of time-series models. Review of the International Statistical Institute, 28(3), 233-244, 1960.
    """
    r = autocorr(x, nTaps)
    w = levinson(r, nTaps)
    return w
