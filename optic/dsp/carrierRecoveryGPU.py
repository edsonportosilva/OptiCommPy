"""
GPU-based carrier recovery utilities.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bps -- Blind phase search (BPS) carrier phase recovery algorithm
"""
import cupy as cp
from cupyx.scipy.signal import oaconvolve

def bpsGPU(Ei, N, constSymb, B):
    """
    Blind phase search (BPS) algorithm

    Parameters
    ----------
    Ei : complex-valued np.array
        Received constellation symbols.
    N : int
        Half of the 2*N+1 average window.
    constSymb : complex-valued np.array
        Complex-valued constellation.
    B : int
        number of test phases.

    Returns
    -------
    θ : real-valued np.array
        Time-varying estimated phase-shifts.

    References
    ----------
    [1] T. Pfau, S. Hoffmann, e R. Noé, “Hardware-efficient coherent digital receiver concept with feedforward carrier recovery for M-QAM constellations”, Journal of Lightwave Technology, vol. 27, nº 8, p. 989–999, 2009, doi: 10.1109/JLT.2008.2010511.
    """
    Ei = cp.asarray(Ei)
    constSymb = cp.asarray(constSymb) 
    
    ϕ_test = cp.arange(0, B) * (cp.pi / 2) / B # test phases
    kernel = cp.ones((2*N+1, 1, 1)) 

    nModes = Ei.shape[1]
    zeroPad = cp.zeros((N, nModes))

    Ei = cp.concatenate(
        (zeroPad, Ei, zeroPad)
    ) # pad start and end of the signal with zeros

    Ei_rotated = Ei[:, :, cp.newaxis] * cp.exp(1j * ϕ_test)[None, None, :]

    dist = cp.absolute(cp.subtract(
        Ei_rotated[:, :, :, None], constSymb[None, None, None, :])) ** 2

    dmin = cp.min(dist, axis=3)
    sumDmin = oaconvolve(dmin, kernel, mode="valid")

    indRot = cp.argmin(sumDmin, axis=2)

    return cp.asnumpy(ϕ_test[indRot])