""" Testing Siemens CSA header reader
"""
import os
from os.path import join as pjoin
import sys
import struct

import numpy as np

from .. import csareader as csa
from .. import dwiparams as dwp

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from ...testing import parametric, IO_DATA_PATH

from .test_dicomwrappers import DATA

CSA2_B0 = open(pjoin(IO_DATA_PATH, 'csa2_b0.bin')).read()
CSA2_B1000 = open(pjoin(IO_DATA_PATH, 'csa2_b1000.bin')).read()


@parametric
def test_csa_header_read():
    hdr = csa.get_csa_header(DATA, 'image')
    yield assert_equal(hdr['n_tags'],83)
    yield assert_equal(csa.get_csa_header(DATA,'series')['n_tags'],65)
    yield assert_raises(ValueError, csa.get_csa_header, DATA,'xxxx')
    yield assert_true(csa.is_mosaic(hdr))
    

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
def test_csa_params():
    for csa_str in (CSA2_B0, CSA2_B1000):
        csa_info = csa.read(csa_str)
        n_o_m = csa.get_n_mosaic(csa_info)
        yield assert_equal(n_o_m, 48)
        snv = csa.get_slice_normal(csa_info)
        yield assert_equal(snv.shape, (3,))
        yield assert_true(np.allclose(1, 
                np.sqrt((snv * snv).sum())))
        amt = csa.get_acq_mat_txt(csa_info)
        yield assert_equal(amt, '128p*128')
    csa_info = csa.read(CSA2_B0)
    b_matrix = csa.get_b_matrix(csa_info)
    yield assert_equal(b_matrix, None)
    b_value = csa.get_b_value(csa_info)
    yield assert_equal(b_value, 0)
    g_vector = csa.get_g_vector(csa_info)
    yield assert_equal(g_vector, None)
    csa_info = csa.read(CSA2_B1000)
    b_matrix = csa.get_b_matrix(csa_info)
    yield assert_equal(b_matrix.shape, (3,3))
    # check (by absence of error) that the B matrix is positive
    # semi-definite. 
    q = dwp.B2q(b_matrix)
    b_value = csa.get_b_value(csa_info)
    yield assert_equal(b_value, 1000)
    g_vector = csa.get_g_vector(csa_info)
    yield assert_equal(g_vector.shape, (3,))
    yield assert_true(
        np.allclose(1, np.sqrt((g_vector * g_vector).sum())))


@parametric
def test_ice_dims():
    ex_dims0 = ['X', '1', '1', '1', '1', '1', '1',
                '48', '1', '1', '1', '1', '201']
    ex_dims1 = ['X', '1', '1', '1', '2', '1', '1',
               '48', '1', '1', '1', '1', '201']
    for csa_str, ex_dims in ((CSA2_B0, ex_dims0),
                             (CSA2_B1000, ex_dims1)):
        csa_info = csa.read(csa_str)
        yield assert_equal(csa.get_ice_dims(csa_info),
                           ex_dims)
    yield assert_equal(csa.get_ice_dims({}), None)
