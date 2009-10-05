''' Testing trackvis module '''

from StringIO import StringIO

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

import dipy.io.trackvis as tv


def test_write():
    streams = []
    out_f = StringIO()
    tv.write(out_f, {}, [])
    yield assert_equal, out_f.getvalue(), tv.empty_header().tostring()
    out_f.truncate(0)
    # Write something not-default
    tv.write(out_f, {'id_string':'TRACKb'}, [])
    # read it back
    out_f.seek(0)
    hdr, streams, endian = tv.read(out_f)
    yield assert_equal, hdr['id_string'], 'TRACKb'
    

def test_empty_header():
    for endian in '<>':
        hdr = tv.empty_header(endian)
        yield assert_equal, hdr['id_string'], 'TRACK'
        yield assert_equal, hdr['version'], 1
        yield assert_equal, hdr['hdr_size'], 1000
    hdr_endian = tv.endian_codes[tv.empty_header().dtype.byteorder]
    yield assert_equal, hdr_endian, tv.native_code
