""" Testing reading DICOM files

"""

import os
from os.path import join as pjoin
import gzip

import numpy as np

import dicom

import dipy.io.dicomreaders as didr
import dipy.io.dicomwrappers as didw
import dipy.io.csareader as csar

from dipy.core.geometry import vector_norm

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric

data_path = pjoin(os.path.dirname(__file__), 'data')

data_file = pjoin(data_path, 'siemens_dwi_1000.dcm.gz')

data = dicom.read_file(gzip.open(data_file))

# this affine from our converted image was shown to match our image
# spatially with an image from SPM DICOM conversion. We checked the
# matching with SPM check reg.
expected_affine = np.array(
    [[ -1.796875, 0, 0, 115],
     [0, -1.79684984, -0.01570896, 135.028779],
     [0, -0.00940843750, 2.99995887, -78.710481],
     [0, 0, 0, 1]])

# from Guys and Matthew's SPM code, with Y flip reversed
expected_params = [992.05050247, (0.99997450,
                                  0.00507649,
                                  -0.005023611)]


@parametric
def test_make_wrapper():
    # test wrapper from data, wrapper from file
    for dw in (didw.make_wrapper(data),
               didw.wrapper_from_file(data_file)):
        yield assert_equal(dw.get('InstanceNumber'), 2)
        yield assert_equal(dw.get('AcquisitionNumber'), 2)
        yield assert_raises(KeyError, dw.__getitem__, 'not an item')
        yield assert_true(dw.is_mosaic)
        yield assert_array_almost_equal(
            np.dot(didr.DPCS_TO_TAL, dw.affine),
            expected_affine)


@parametric
def test_wrappers():
    # test direct wrapper calls
    # first with empty data
    klasses = (didw.Wrapper,
               didw.SiemensWrapper,
               didw.MosaicWrapper)
    for klass in klasses:
        dw = klass()
        yield assert_equal(dw.get('InstanceNumber'), None)
        yield assert_equal(dw.get('AcquisitionNumber'), None)
        yield assert_raises(KeyError, dw.__getitem__, 'not an item')
        yield assert_raises(didw.WrapperError, dw.get_data)
        yield assert_raises(didw.WrapperError, getattr, dw, 'affine')
    for klass in (didw.Wrapper, didw.SiemensWrapper):
        dw = klass()
        yield assert_false(dw.is_mosaic)
    for maker in klasses + (didw.make_wrapper,):
        dw = maker(data)
        yield assert_equal(dw.get('InstanceNumber'), 2)
        yield assert_equal(dw.get('AcquisitionNumber'), 2)
        yield assert_raises(KeyError, dw.__getitem__, 'not an item')
    for maker in (didw.MosaicWrapper, didw.make_wrapper):
        yield assert_true(dw.is_mosaic)


@parametric
def test_read_dwi():
    img = didr.mosaic_to_nii(data)
    arr = img.get_data()
    yield assert_equal(arr.shape, (128,128,48))
    yield assert_array_almost_equal(img.get_affine(), expected_affine)


@parametric
def test_csa_header_read():
    hdr = csar.get_csa_header(data, 'image')
    yield assert_equal(hdr['n_tags'],83)
    yield assert_equal(csar.get_csa_header(data,'series')['n_tags'],65)
    yield assert_raises(ValueError, csar.get_csa_header, data,'xxxx')
    yield assert_true(csar.is_mosaic(hdr))
    

@parametric
def test_dwi_params():
    dw = didw.make_wrapper(data)
    b_matrix = dw.b_matrix
    yield assert_equal(b_matrix.shape, (3,3))
    q = dw.q_vector
    b = vector_norm(q)
    g = q / b
    yield assert_array_almost_equal(b, expected_params[0])
    yield assert_array_almost_equal(g, expected_params[1])


@parametric
def test_read_dwis():
    data, aff, bs, gs = didr.read_mosaic_dwi_dir(data_path, '*.dcm.gz')
    yield assert_equal(data.ndim, 4)
    yield assert_equal(aff.shape, (4,4))
    yield assert_equal(bs.shape, (2,))
    yield assert_equal(gs.shape, (2,3))
    yield assert_raises(IOError, didr.read_mosaic_dwi_dir, 'improbable')
