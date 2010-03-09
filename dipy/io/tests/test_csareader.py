""" Testing Siemens CSA header reader
"""
import os
from os.path import join as pjoin
import struct

import numpy as np

import dipy.io.csareader as csa

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric


data_path = pjoin(os.path.dirname(__file__), 'data')

CSA2_B0 = open(pjoin(data_path, 'csa2_b0.bin')).read()
CSA2_B1000 = open(pjoin(data_path, 'csa2_b1000.bin')).read()


@parametric
def test_csas0():
    for csa_str in (CSA2_B0, CSA2_B1000):
        csa_info = csa.read(csa_str)
        yield assert_equal(csa_info['type'], 2)
        yield assert_equal(csa_info['n_tags'], 83)
        tags = csa_info['tags']
        yield assert_equal(len(tags), 83)
        n_o_m = tags['NumberOfImagesInMosaic']
        yield assert_equal(n_o_m['items'], [48])
    csa_info = csa.read(CSA2_B1000)
    b_matrix = csa_info['tags']['B_matrix']
    yield assert_equal(len(b_matrix['items']), 6)
    b_value = csa_info['tags']['B_value']
    yield assert_equal(b_value['items'], [1000])


@parametric
def test_unpacker():
    s = '1234\x00\x01'
    le_int, = struct.unpack('<h', '\x00\x01')
    be_int, = struct.unpack('>h', '\x00\x01')
    up_str = csa.Unpacker(s, endian='<')
    yield assert_equal(up_str.read(4), '1234')
    up_str.ptr = 0
    yield assert_equal(up_str.unpack('4s'), ('1234',))
    yield assert_equal(up_str.unpack('h'), (le_int,))
    up_str = csa.Unpacker(s, endian='>')
    yield assert_equal(up_str.unpack('4s'), ('1234',))
    yield assert_equal(up_str.unpack('h'), (be_int,))
