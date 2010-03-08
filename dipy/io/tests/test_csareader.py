""" Testing Siemens CSA header reader
"""
import os
from os.path import join as pjoin

import numpy as np

import dipy.io.csareader as csa

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric


data_path = pjoin(os.path.dirname(__file__), 'data')

CSA2_B0 = pjoin(data_path, 'csa2_b0.bin')
CSA2_B1000 = pjoin(data_path, 'csa2_b1000.bin')


@parametric
def test_csa():
    csa_f = open(CSA2_B0, 'rb')
    hdr, tags = csa.read(csa_f)
    yield assert_equal(hdr['type'], 2)
    yield assert_equal(hdr['n_tags'], 83)
    yield assert_equal(len(tags), 83)
    
