""" Testing Siemens CSA header reader
"""
import sys
import struct

from dipy.io.structreader import Unpacker

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric


@parametric
def test_unpacker():
    s = '1234\x00\x01'
    le_int, = struct.unpack('<h', '\x00\x01')
    be_int, = struct.unpack('>h', '\x00\x01')
    if sys.byteorder == 'little':
        native_int = le_int
        swapped_int = be_int
        native_code = '<'
        swapped_code = '>'
    else:
        native_int = be_int
        swapped_int = le_int
        native_code = '>'
        swapped_code = '<'
    up_str = Unpacker(s, endian='<')
    yield assert_equal(up_str.read(4), '1234')
    up_str.ptr = 0
    yield assert_equal(up_str.unpack('4s'), ('1234',))
    yield assert_equal(up_str.unpack('h'), (le_int,))
    up_str = Unpacker(s, endian='>')
    yield assert_equal(up_str.unpack('4s'), ('1234',))
    yield assert_equal(up_str.unpack('h'), (be_int,))
    # now test conflict of endian
    up_str = Unpacker(s, ptr=4, endian='>')    
    yield assert_equal(up_str.unpack('<h'), (le_int,))
    up_str = Unpacker(s, ptr=4, endian=swapped_code)
    yield assert_equal(up_str.unpack('h'), (swapped_int,))
    up_str.ptr = 4
    yield assert_equal(up_str.unpack('<h'), (le_int,))
    up_str.ptr = 4
    yield assert_equal(up_str.unpack('>h'), (be_int,))
    up_str.ptr = 4
    yield assert_equal(up_str.unpack('@h'), (native_int,))
    
