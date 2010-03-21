''' Testing trackvis module '''

from StringIO import StringIO

import numpy as np

import dipy.io.trackvis as tv

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric


def test_write():
    streams = []
    out_f = StringIO()
    tv.write(out_f, [], {})
    yield assert_equal, out_f.getvalue(), tv.empty_header().tostring()
    out_f.truncate(0)
    # Write something not-default
    tv.write(out_f, [], {'id_string':'TRACKb'})
    # read it back
    out_f.seek(0)
    streams, hdr = tv.read(out_f)
    yield assert_equal, hdr['id_string'], 'TRACKb'
    # check that we can pass none for the header
    out_f.truncate(0)
    tv.write(out_f, [])
    out_f.truncate(0)
    tv.write(out_f, [], None)
    # check that we check input values
    out_f.truncate(0)
    yield (assert_raises, tv.HeaderError,
           tv.write, out_f, [],{'id_string':'not OK'})
    yield (assert_raises, tv.HeaderError,
           tv.write, out_f, [],{'version':2})
    yield (assert_raises, tv.HeaderError,
           tv.write, out_f, [],{'hdr_size':0})


def streams_equal(stream1, stream2):
    if not np.all(stream1[0] == stream1[0]):
        return False
    if stream1[1] is None:
        if not stream2[1] is None:
            return false
    if stream1[2] is None:
        if not stream2[2] is None:
            return false
    if not np.all(stream1[1] == stream1[1]):
        return False
    if not np.all(stream1[2] == stream1[2]):
        return False
    return True


def streamlist_equal(streamlist1, streamlist2):
    if len(streamlist1) != len(streamlist2):
        return False
    for s1, s2 in zip(streamlist1, streamlist2):
        if not streams_equal(s1, s2):
            return False
    return True


def test_round_trip():
    out_f = StringIO()
    xyz0 = np.tile(np.arange(5).reshape(5,1), (1, 3))
    xyz1 = np.tile(np.arange(5).reshape(5,1) + 10, (1, 3))
    streams = [(xyz0, None, None), (xyz1, None, None)]
    tv.write(out_f, streams, {})
    out_f.seek(0)
    streams2, hdr = tv.read(out_f)
    yield assert_true, streamlist_equal(streams, streams2)
        

@parametric
def test_empty_header():
    for endian in '<>':
        hdr = tv.empty_header(endian)
        yield assert_equal(hdr['id_string'], 'TRACK')
        yield assert_equal(hdr['version'], 1)
        yield assert_equal(hdr['hdr_size'], 1000)
        yield assert_array_equal(
            hdr['image_orientation_patient'],
            [0,0,0,0,0,0])
    hdr_endian = tv.endian_codes[tv.empty_header().dtype.byteorder]
    yield assert_equal(hdr_endian, tv.native_code)


@parametric
def test_get_affine():
    hdr = tv.empty_header()
    # default header gives useless affine
    yield assert_array_equal(tv.aff_from_hdr(hdr),
                             np.diag([0,0,0,1]))
    hdr['voxel_size'] = 1
    yield assert_array_equal(tv.aff_from_hdr(hdr),
                             np.diag([0,0,0,1]))
    # DICOM direction cosines
    hdr['image_orientation_patient'] = [1,0,0,0,1,0]
    yield assert_array_equal(tv.aff_from_hdr(hdr),
                             np.diag([-1,-1,1,1]))
    # RAS direction cosines
    hdr['image_orientation_patient'] = [-1,0,0,0,-1,0]
    yield assert_array_equal(tv.aff_from_hdr(hdr),
                             np.eye(4))
    # translations
    hdr['origin'] = [1,2,3]
    exp_aff = np.eye(4)
    exp_aff[:3,3] = [-1,-2,3]
    yield assert_array_equal(tv.aff_from_hdr(hdr),
                             exp_aff)
    # mappings work too
    d = {'voxel_size': np.array([1,2,3]),
         'image_orientation_patient': np.array([1,0,0,0,1,0]),
         'origin': np.array([10,11,12])}
    aff = tv.aff_from_hdr(d)


@parametric
def test_aff_to_hdr():
    hdr = {}
    affine = np.diag([1,2,3,1])
    affine[:3,3] = [10,11,12]
    tv.aff_to_hdr(affine, hdr)
    yield assert_array_almost_equal(tv.aff_from_hdr(hdr), affine)
    # put flip into affine
    aff2 = affine.copy()
    aff2[:,2] *=-1
    tv.aff_to_hdr(aff2, hdr)
    yield assert_array_almost_equal(tv.aff_from_hdr(hdr), aff2)


@parametric
def test_tv_class():
    tvf = tv.TrackvisFile([])
    yield assert_equal(tvf.streamlines, [])
    yield assert_true(isinstance(tvf.header, np.ndarray))
    yield assert_equal(tvf.endianness, tv.native_code)
    yield assert_equal(tvf.filename, None)
    out_f = StringIO()
    tvf.to_file(out_f)
    yield assert_equal(out_f.getvalue(), tv.empty_header().tostring())
    out_f.truncate(0)
    # Write something not-default
    tvf = tv.TrackvisFile([], {'id_string':'TRACKb'})
    tvf.to_file(out_f)
    # read it back
    out_f.seek(0)
    tvf_back = tv.TrackvisFile.from_file(out_f)
    yield assert_equal(tvf_back.header['id_string'], 'TRACKb')
    # check that we check input values
    out_f.truncate(0)
    yield assert_raises(tv.HeaderError,
                        tv.TrackvisFile,
                        [],{'id_string':'not OK'})
    yield assert_raises(tv.HeaderError,
                        tv.TrackvisFile,
                        [],{'version': 2})
    yield assert_raises(tv.HeaderError,
                        tv.TrackvisFile,
                        [],{'hdr_size':0})
    affine = np.diag([1,2,3,1])
    affine[:3,3] = [10,11,12]
    tvf.set_affine(affine)
    yield assert_true(np.all(tvf.get_affine() == affine))
