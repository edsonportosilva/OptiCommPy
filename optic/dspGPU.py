import cupy as cp

def firFilter(h, x):
    '''
    Performs FIR filtering and compensates for filter delay
    (assuming the impulse response is symmetric)
    
    Parameters
    ----------
    h : ndarray
        Coefficients of the FIR filter.
    x : ndarray
        Input signal.
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
        
    # fft in CuPy uses only complex64
    prec = cp.complex64
    x_ = cp.asarray(x).astype(prec)
    h_ = cp.asarray(h).astype(prec)
    y_ = x_.copy()
    
    for n in range(0, nModes):
        y_[:, n] = cp.convolve(x_[:, n], h_, mode='same')
    
    y = cp.asnumpy(y_)
    
    if y.shape[1] == 1:
        y = y[:, 0]

    return y