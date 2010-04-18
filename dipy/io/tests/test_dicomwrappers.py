""" Testing DICOM wrappers
"""

from os.path import join as pjoin
import gzip

import numpy as np

import dicom

from .. import dicomwrappers as didw
from .. import dicomreaders as didr
from ...core.geometry import vector_norm

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from ...testing import parametric, IO_DATA_PATH

DATA_FILE = pjoin(IO_DATA_PATH, 'siemens_dwi_1000.dcm.gz')
DATA = dicom.read_file(gzip.open(DATA_FILE))
DATA_FILE_B0 = pjoin(IO_DATA_PATH, 'siemens_dwi_0.dcm.gz')

# this affine from our converted image was shown to match our image
# spatially with an image from SPM DICOM conversion. We checked the
# matching with SPM check reg.  We have flipped the first and second
# rows to allow for rows, cols tranpose. 
EXPECTED_AFFINE = np.array(
    [[ -1.796875, 0, 0, 115],
     [0, -1.79684984, -0.01570896, 135.028779],
     [0, -0.00940843750, 2.99995887, -78.710481],
     [0, 0, 0, 1]])[:,[1,0,2,3]]

# from Guys and Matthew's SPM code, undoing SPM's Y flip, and swapping
# first two values in vector, to account for data rows, cols difference.
EXPECTED_PARAMS = [992.05050247, (0.00507649,
                                  0.99997450,
                                  -0.005023611)]


@parametric
def test_wrappers():
    # test direct wrapper calls
    # first with empty data
    for maker, kwargs in ((didw.Wrapper,{}),
                          (didw.SiemensWrapper, {}),
                          (didw.MosaicWrapper, {'n_mosaic':10})):
        dw = maker(**kwargs)
        yield assert_equal(dw.get('InstanceNumber'), None)
        yield assert_equal(dw.get('AcquisitionNumber'), None)
        yield assert_raises(KeyError, dw.__getitem__, 'not an item')
        yield assert_raises(didw.WrapperError, dw.get_data)
        yield assert_raises(didw.WrapperError, dw.get_affine)
    for klass in (didw.Wrapper, didw.SiemensWrapper):
        dw = klass()
        yield assert_false(dw.is_mosaic)
    for maker in (didw.wrapper_from_data,
                  didw.Wrapper,
                  didw.SiemensWrapper,
                  didw.MosaicWrapper
                  ):
        dw = maker(DATA)
        yield assert_equal(dw.get('InstanceNumber'), 2)
        yield assert_equal(dw.get('AcquisitionNumber'), 2)
        yield assert_raises(KeyError, dw.__getitem__, 'not an item')
    for maker in (didw.MosaicWrapper, didw.wrapper_from_data):
        yield assert_true(dw.is_mosaic)


@parametric
def test_wrapper_from_data():
    # test wrapper from data, wrapper from file
    for dw in (didw.wrapper_from_data(DATA),
               didw.wrapper_from_file(DATA_FILE)):
        yield assert_equal(dw.get('InstanceNumber'), 2)
        yield assert_equal(dw.get('AcquisitionNumber'), 2)
        yield assert_raises(KeyError, dw.__getitem__, 'not an item')
        yield assert_true(dw.is_mosaic)
        yield assert_array_almost_equal(
            np.dot(didr.DPCS_TO_TAL, dw.get_affine()),
            EXPECTED_AFFINE)


@parametric
def test_dwi_params():
    dw = didw.wrapper_from_data(DATA)
    b_matrix = dw.b_matrix
    yield assert_equal(b_matrix.shape, (3,3))
    q = dw.q_vector
    b = vector_norm(q)
    g = q / b
    yield assert_array_almost_equal(b, EXPECTED_PARAMS[0])
    yield assert_array_almost_equal(g, EXPECTED_PARAMS[1])


@parametric
def test_vol_matching():
    # make the Siemens wrapper, check it compares True against itself
    dw_siemens = didw.wrapper_from_data(DATA)
    yield assert_true(dw_siemens.is_mosaic)
    yield assert_true(dw_siemens.is_csa)
    yield assert_true(dw_siemens.is_same_series(dw_siemens))
    # make plain wrapper, compare against itself
    dw_plain = didw.Wrapper(DATA)
    yield assert_false(dw_plain.is_mosaic)
    yield assert_false(dw_plain.is_csa)
    yield assert_true(dw_plain.is_same_series(dw_plain))
    # specific vs plain wrapper compares False, because the Siemens
    # wrapper has more non-empty information
    yield assert_false(dw_plain.is_same_series(dw_siemens))
    # and this should be symmetric
    yield assert_false(dw_siemens.is_same_series(dw_plain))
    # we can even make an empty wrapper.  This compares True against
    # itself but False against the others
    dw_empty = didw.Wrapper()
    yield assert_true(dw_empty.is_same_series(dw_empty))
    yield assert_false(dw_empty.is_same_series(dw_plain))
    yield assert_false(dw_plain.is_same_series(dw_empty))
    # Just to check the interface, make a pretend signature-providing
    # object.
    class C(object):
        series_signature = {}
    yield assert_true(dw_empty.is_same_series(C()))


@parametric
def test_slice_indicator():
    dw_0 = didw.wrapper_from_file(DATA_FILE_B0)
    dw_1000 = didw.wrapper_from_data(DATA)
    z = dw_0.slice_indicator
    yield assert_false(z is None)
    yield assert_equal(z, dw_1000.slice_indicator)
    dw_empty = didw.Wrapper()
    yield assert_true(dw_empty.slice_indicator is None)
