""" Testing DICOM wrappers

DICOM wrappers are objects that contain things like DICOM datasets.

They are to make it easier to add new fields from processing -
particularly the Siemens private header stuff.

They define a standard set of fields that we will use, where None is the
default.  We access the fields by using the ``.fields`` attribute, which
is a mapping.

There are some attributes (not fields), such as:

* is_mosaic

The object also adds methods like:

* get_data() - get scaled data from pixel array
* get_affine() - get constructed affine

"""

from os.path import join as pjoin, dirname
import gzip

import numpy as np

import dicom

from dipy.io.dicomwrappers import Wrapper, WrapperError

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import data_path, parametric

data_path = pjoin(dirname(__file__), 'data')

data_file = pjoin(data_path, 'siemens_dwi_1000.dcm.gz')
data = dicom.read_file(gzip.open(data_file))


@parametric
def test_wrappers():
    # make test object
    dw = Wrapper()
    yield assert_equal(dw.get('InstanceNumber'), None)
    yield assert_equal(dw.get('AcquisitionNumber'), None)
    yield assert_raises(KeyError, dw.__getitem__, 'not an item')
    yield assert_false(dw.is_mosaic)
    yield assert_raises(KeyError, dw.get_data)
    yield assert_raises(WrapperError, getattr, dw, 'affine')
    dw = Wrapper(data)
    yield assert_equal(dw.get('InstanceNumber'), 2)
    yield assert_equal(dw.get('AcquisitionNumber'), 2)
    yield assert_raises(KeyError, dw.__getitem__, 'not an item')
    yield assert_true(dw.is_mosaic)
    
