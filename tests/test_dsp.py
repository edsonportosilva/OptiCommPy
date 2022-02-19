# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 12:58:47 2022

@author: edson
"""

from optic.dsp import finddelay
import numpy as np


def test_finddelay_for_arrays_of_real_values():
    delay = 35

    a = np.arange(0, 100)
    b = np.roll(a, -delay)

    assert delay == finddelay(a, b)


def test_finddelay_for_arrays_of_complex_values():
    delay = 57

    a = np.arange(0, 100) + 1j * np.arange(0, 100)
    b = np.roll(a, -delay)

    assert delay == finddelay(a, b)
