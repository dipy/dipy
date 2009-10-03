""" Read and write trackvis files
"""
import struct

import numpy as np

from .utils import native_code, swapped_code, endian_codes


# Definition of trackvis header structure.
# See http://www.trackvis.org/docs/?subsect=fileformat
# See http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
header_dtd = [
    ('id_string', 'S6'),
    ('dim', 'h', 3),
    ('voxel_size', 'f4', 3),
    ('origin', 'f4', 3),
    ('n_scalars', 'h'),
    ('scalar_name', 'S20', 10),
    ('n_properties', 'h'),
    ('property_name', 'S20', 10),
    ('reserved', 'S508'),
    ('voxel_order', 'S4'),
    ('pad2', 'S4'),
    ('image_orientation_patient', 'f4', 6),
    ('pad1', 'S2'),
    ('invert_x', 'S1'),
    ('invert_y', 'S1'),
    ('invert_z', 'S1'),
    ('swap_xy', 'S1'),
    ('swap_yz', 'S1'),
    ('swap_zx', 'S1'),
    ('n_count', 'i4'),
    ('version', 'i4'),
    ('hdr_size', 'i4'),
    ]

# Full header numpy dtype
header_dtype = np.dtype(header_dtd)


class HeaderError(Exception):
    pass


def read(fileobj):
    hdr_str = fileobj.read(header_dtype.itemsize)
    hdr = np.ndarray(shape=(),
                     dtype=header_dtype,
                     buffer=hdr_str)
    if hdr['hdr_size'] == 1000:
        endianness = native_code
    else:
        hdr = hdr.newbyteorder()
        if hdr['hdr_size'] != 1000:
            raise HeaderError('Invalid hdr_size of %s'
                              % hdr['hdr_size'])
        endianness = swapped_code
    n_s = hdr['n_scalars']
    n_p = hdr['n_properties']
    point_dtype = np.dtype(
        [('x', 'f4'),
         ('y','f4'),
         ('z','f4'),
         ('scalars', 'f4', n_s)]).newbyteorder(endianness)
    pt_size = point_dtype.itemsize
    property_dtype = np.dtype('f4').newbyteorder(endianness)
    ps_size = property_dtype.itemsize * n_p
    i_fmt = endianness + 'i'
    streamlines = []
    stream_count = hdr['n_count']
    if stream_count < 0:
        raise HeaderError('Unexpeted negative n_count')
    n_streams = 0
    while(True):
        n_str = fileobj.read(4)
        if len(n_str) < 4:
            if stream_count:
                raise HeaderError(
                    'Expecting %s points, found only %s' % (
                            stream_count, n_streams))
            break
        n_pts = struct.unpack(i_fmt, n_str)[0]        
        pts_str = fileobj.read(n_pts * pt_size)
        pts = np.ndarray(
            shape = (n_pts,),
            dtype = point_dtype,
            buffer = pts_str)
        if n_p:
            ps_str = fileobj.read(ps_size)
            ps = np.ndarray(
                shape = (n_p,),
                dtype = property_dtype,
                buffer = ps_str)
        else:
            ps = np.array([], dtype='f4')
        streamlines.append((pts, ps))
        n_streams += 1
        # deliberately misses case where stream_count is 0
        if n_streams == stream_count:
            break
    return hdr, streamlines, endianness


def write(fileobj, hdr, streamlines, endianness=native_code):
    endianness = endian_codes[endianness]
    if isinstance(hdr, np.ndarray):
        if endian_codes[hdr.dtype.byteorder] == endianness:
            hdr = hdr.copy()
        else:
            hdr = hdr.byteswap().newbyteorder()
    else:
        mapping = hdr
        hdr = empty_header(endianness)
        for key, value in mapping.items():
            hdr[key] = value
    stream_count = len(streamlines)
    hdr['n_count'] = stream_count
    pts0, ps0 = streamlines[0]
    # calculate number of scalars
    n_s = pts0['scalars'].size
    hdr['n_scalars'] = n_s
    # calculate number of properties
    n_p = ps0.size
    hdr['n_properties'] = n_p
    fileobj.write(hdr.tostring())
    i_fmt = endianness + 'i'
    for pts, props in streamlines:
        n_pts = pts.size
        fileobj.write(struct.pack(i_fmt, n_pts))
        if endian_codes[pts.dtype.byteorder] != endianness:
            pts = pts.byteswap().newbyteorder()
        fileobj.write(pts.tostring())
        if n_p:
            if endian_codes[props.dtype.byteorder] != endianness:
                props = props.byteswap().newbyteorder()
            fileobj.write(props.tostring())


def empty_header(endianness=None):
    ''' Empty trackvis header '''
    dt = header_dtype
    if endianness:
        dt = dt.newbyteorder(endianness)
    return np.zeros((), dtype=dt)
