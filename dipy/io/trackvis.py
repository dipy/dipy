""" Read and write trackvis files
"""
import struct

import numpy as np

from .utils import native_code, swapped_code, endian_codes, \
    allopen, rec2dict

from dipy.core.streamlines import StreamLine

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


class DataError(Exception):
    pass


def read(fileobj):
    ''' Read trackvis file, return header, streamlines, endianness

    Parameters
    ----------
    fileobj : string or file-like object
       If string, a filename; otherwise an open file-like object
       pointing to trackvis file (and ready to read from the beginning
       of the trackvis header data)

    Returns
    -------
    hdr : structured array
       structured array with trackvis header fields
    streamlines : sequence
       sequence of :class:`streamLine`
    endianness : {'<', '>'}
       Endianness of read header, '<' is little-endian, '>' is
       big-endian
    '''
    fileobj = allopen(fileobj, mode='rb')
    hdr_str = fileobj.read(header_dtype.itemsize)
    hdr = np.ndarray(shape=(),
                     dtype=header_dtype,
                     buffer=hdr_str)
    if str(hdr['id_string'])[:5] != 'TRACK':
        raise HeaderError('Expecting TRACK as first '
                          '5 characaters of id_string')
    if hdr['version'] > 1:
        raise HeaderError('Reader only supports version 1')
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
    f4dt = np.dtype(endianness + 'f4')
    pt_cols = 3 + n_s
    pt_size = f4dt.itemsize * pt_cols
    ps_size = f4dt.itemsize * n_p
    i_fmt = endianness + 'i'
    streamlines = []
    stream_count = hdr['n_count']
    if stream_count < 0:
        raise HeaderError('Unexpected negative n_count')
    n_streams = 0
    # For case where there are no scalars or no properties
    scalars = None
    ps = None
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
            shape = (n_pts, pt_cols),
            dtype = f4dt,
            buffer = pts_str)
        if n_p:
            ps_str = fileobj.read(ps_size)
            ps = np.ndarray(
                shape = (n_p,),
                dtype = f4dt,
                buffer = ps_str)
        xyz = pts[:,:3]
        if n_s:
            scalars = pts[:,3:]
        streamlines.append(StreamLine(xyz, scalars, ps))
        n_streams += 1
        # deliberately misses case where stream_count is 0
        if n_streams == stream_count:
            break
    return hdr, streamlines, endianness


def write(fileobj, hdr_mapping, streamlines, endianness=None):
    ''' Write header and `streamlines` to trackvis file `fileobj` 

    The parameters from the streamlines override conflicting parameters
    in the `hdr_mapping` information.  In particular, the number of
    streamlines, the number of scalars, and the number of properties are
    written according to `streamlines` rather than `hdr_mapping`.

    Parameters
    ----------
    fileobj : filename or file-like
       If filename, open file as 'wb', otherwise `fileobj` should be an
       open file-like object, with a ``write`` method.
    hdr_mapping : mapping
       Information for filling header fields.  Can be something
       dict-like (implementing ``items``) or a structured numpy array
    streamlines : sequence
       sequence of :class:`StreamLine`
    endianness : {None, '<', '>'}, optional
       Endianness of file to be written.  '<' is little-endian, '>' is
       big-endian.  None (the default) is to use the endianness of the
       `streamlines` data.

    Returns
    -------
    None
    
    '''
    # First work out guessed endianness from input data
    stream_count = len(streamlines)
    if endianness is None:
        if stream_count:
            endianness = streamlines[0][0].dtype.byteorder
        else:
            endianness = native_code
    endianness = endian_codes[endianness]
    # fill in a new header from mapping-like
    if isinstance(hdr_mapping, np.ndarray):
        hdr_mappping = rec2dict(hdr_mapping)
    hdr = empty_header(endianness)
    for key, value in hdr_mapping.items():
        hdr[key] = value
    # put calculated data into header
    hdr['n_count'] = stream_count
    if stream_count:
        pts0, ps0 = streamlines[0]
        # calculate number of scalars
        n_s = pts0['scalars'].size
        hdr['n_scalars'] = n_s
        # calculate number of properties
        n_p = ps0.size
        hdr['n_properties'] = n_p
    # write header
    fileobj = allopen(fileobj, mode='wb')
    fileobj.write(hdr.tostring())
    if not stream_count:
        return
    point_dtype, property_dtype = _pts_props_dtypes(n_s, endianness)
    point_dtype_fields = point_dtype.fields
    property_dtype_fields = property_dtype.fields
    i_fmt = endianness + 'i'
    for pts, props in streamlines:
        n_pts = pts.size
        fileobj.write(struct.pack(i_fmt, n_pts))
        # This call ensures that the data are 32-bit floats, and that
        # the endianness is OK.
        if pts.dtype.fields != point_dtype_fields:
            pts = pts.astype(point_dtype)
        fileobj.write(pts.tostring())
        if n_p:
            if props.dtype.fields != property_dtype_fields:
                props = props.astype(point_dtype)
            fileobj.write(props.tostring())


def _pts_props_dtypes(n_scalars, endianness):
    ''' point, property dtypes given number of scalars, endianness
    '''
    point_dtype = np.dtype(
        [('x','f4'),
         ('y','f4'),
         ('z','f4'),
         ('scalars', 'f4', n_scalars)]).newbyteorder(endianness)
    property_dtype = np.dtype('f4').newbyteorder(endianness)
    return point_dtype, property_dtype
    

def empty_header(endianness=None):
    ''' Empty trackvis header
    
    Parameters
    ----------
    endianness : {'<','>'}, optional
       Endianness of empty header to return. Default is native endian.

    Returns
    -------
    hdr : structured array
       structured array containing empty trackvis header

    Examples
    --------
    >>> hdr = empty_header()
    >>> print hdr['version']
    1
    >>> print hdr['id_string']
    TRACK
    >>> endian_codes[hdr['version'].dtype.byteorder] == native_code
    True
    >>> hdr = empty_header(swapped_code)
    >>> endian_codes[hdr['version'].dtype.byteorder] == swapped_code
    True
    '''
    dt = header_dtype
    if endianness:
        dt = dt.newbyteorder(endianness)
    hdr = np.zeros((), dtype=dt)
    hdr['id_string'] = 'TRACK'
    hdr['version'] = 1
    hdr['hdr_size'] = 1000
    return hdr
