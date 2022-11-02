import cupy as cp

def firFilter(h, x, prec = cp.complex128):
    '''
    Performs FIR filtering and compensates for filter delay
    (assuming the impulse response is symmetric)
    
    Parameters
    ----------
    h : ndarray
        Coefficients of the FIR filter.
    x : ndarray
        Input signal.
    prec: cp.dtype
        Size of the complex representation.
        
    Returns
    -------
    y : ndarray
        Output (filtered) signal.
    '''

    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(len(x), 1)


    nModes = x.shape[1]

    x_ = cp.asarray(x).astype(prec)
    h_ = cp.asarray(h).astype(prec)
    y_ = x_.copy()

    for n in range(nModes):
        y_[:, n] = cp.convolve(x_[:, n], h_, mode='same')

    y = cp.asnumpy(y_)

    if y.shape[1] == 1:
        y = y[:, 0]

    return y