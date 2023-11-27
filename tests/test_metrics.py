import unittest
import numpy as np
from optic.dsp.core import pnorm
from optic.models.channels import awgn
from optic.comm.modulation import modulateGray
from optic.comm.metrics import (
    fastBERcalc,
    calcLLR,
    monteCarloGMI,
    monteCarloMI,
    Qfunc,
    calcEVM,
    theoryBER,
    calcLinOSNR,
    GN_Model_NyquistWDM,
    ASE_NyquistWDM,
    GNmodel_OSNR,
)

class TestCommunicationMetrics(unittest.TestCase):
    
    def setUp(self):
        # Set up common parameters or fixtures if needed
        pass

    def test_fastBERcalc(self):
        # Add test cases for fastBERcalc

        # Run BER vs Ebn0 Monte Carlo simulation in the AWGN channel

        qamOrder  = [4, 16, 64, 256]  # Modulation order

        EbN0dB_  = np.arange(10, 25.5, 0.5)
        BER      = np.zeros((len(EbN0dB_),len(qamOrder)))
        BER_theory = np.zeros((len(EbN0dB_),len(qamOrder)))

        for ii, M in enumerate(qamOrder):
        
            for indSNR in range(EbN0dB_.size):

                EbN0dB = EbN0dB_[indSNR]

                # generate random bits
                bitsTx = np.random.randint(2, size=int(np.log2(M)*1e5))    

                # Map bits to constellation symbols
                symbTx = modulateGray(bitsTx, M, 'qam')

                # Normalize symbols energy to 1
                symbTx = pnorm(symbTx) 

                # AWGN channel  
                snrdB  = EbN0dB + 10*np.log10(np.log2(M))
                symbRx = awgn(symbTx, snrdB)

                # BER calculation
                BER[indSNR, ii], _, _ = fastBERcalc(symbRx, symbTx, M, 'qam')
                BER_theory[indSNR, ii] = theoryBER(M, EbN0dB, 'qam')

                if BER[indSNR, ii] == 0:              
                    break

        np.testing.assert_array_almost_equal(BER, BER_theory, decimal=3)

        

    def test_calcLLR(self):
        # Add test cases for calcLLR
        pass

    def test_monteCarlo_MI_and_GMI(self):
        # Run GMI vs SNR Monte Carlo simulation
        qamOrder  = [4, 16, 64]  # Modulation order

        SNR  = np.arange(15, 26, 1)
        MI  = np.zeros((len(SNR),len(qamOrder)))
        GMI  = np.zeros((len(SNR),len(qamOrder)))

        for ii, M in enumerate(qamOrder):            
            for indSNR in range(SNR.size):

                snrdB = SNR[indSNR]

                # generate random bits
                bitsTx   = np.random.randint(2, size=int(np.log2(M)*1e5))    

                # Map bits to constellation symbols
                symbTx = modulateGray(bitsTx, M, 'qam')

                # Normalize symbols energy to 1
                symbTx = pnorm(symbTx) 

                # AWGN channel       
                symbRx = awgn(symbTx, snrdB)

                # GMI estimation
                MI[indSNR, ii] = monteCarloMI(symbRx, symbTx, M, 'qam')
                GMI[indSNR, ii], _  = monteCarloGMI(symbRx, symbTx, M, 'qam')

        np.testing.assert_array_almost_equal(MI, GMI, decimal=1)           
   
    def test_Qfunc(self):
        # Add test cases for Qfunc
        pass

    def test_calcEVM(self):
        # Add test cases for calcEVM
        pass

    def test_theoryBER(self):
        # Add test cases for theoryBER
        pass

    def test_calcLinOSNR(self):
        # Add test cases for calcLinOSNR
        pass

    def test_GN_Model_NyquistWDM(self):
        # Add test cases for GN_Model_NyquistWDM
        pass

    def test_ASE_NyquistWDM(self):
        # Add test cases for ASE_NyquistWDM
        pass

    def test_GNmodel_OSNR(self):
        # Add test cases for GNmodel_OSNR
        pass

if __name__ == '__main__':
    unittest.main()
