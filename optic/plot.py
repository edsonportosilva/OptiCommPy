# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 00:31:26 2021

@author: edson
"""

import matplotlib.pyplot as plt
import numpy as np
from optic.metrics import signal_power

def pconst(x):

    if type(x) == list:
        nSubPts = x[0].shape[1] 
        radius = 1.5*np.sqrt(signal_power(x[0]))
    else:
        nSubPts = x.shape[1]
        radius = 1.5*np.sqrt(signal_power(x))
        
    if nSubPts > 1:
        if nSubPts < 5:
            nCols = nSubPts
            nRows = 1
        elif nSubPts >= 6:
            nCols = np.ceil(nSubPts/2)
            nRows = 2
    
        # Create a Position index
        Position = range(1, nSubPts + 1)
        
        fig = plt.figure()
    
        if type(x) == list:
            for k in range(nSubPts):  
                ax = fig.add_subplot(nRows, nCols, Position[k])
                
                for ind in range(len(x)):
                    ax.plot(x[ind][:,k].real, x[ind][:,k].imag,'.')
                    
                ax.axis('square')           
                ax.grid()
                ax.set_title('mode '+str(Position[k]-1))
                ax.set_xlim(-radius, radius)
                ax.set_ylim(-radius, radius);
        else:
            for k in range(nSubPts):  
                ax = fig.add_subplot(nRows, nCols, Position[k])
                ax.plot(x[:,k].real, x[:,k].imag,'.')
                ax.axis('square')           
                ax.grid()
                ax.set_title('mode '+str(Position[k]-1))
                ax.set_xlim(-radius, radius)
                ax.set_ylim(-radius, radius);
            
    elif nSubPts == 1:
        plt.figure()
        plot(x.real, x.imag,'.')
        plt.axis('square')
        plt.xlim(-radius, radius)
        plt.ylim(-radius, radius);    
       
    plt.show()