""" Read and write trackvis files
"""
import struct

import numpy as np
import numpy.linalg as npl

from .utils import native_code, swapped_code, endian_codes, \
    allopen, rec2dict

# Definition of trackvis header structure.
# See http://www.trackvis.org/docs/?subsect=fileformat
# See http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
header_1_dtd = [
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

# Version 2 adds a 4x4 matrix giving the affine transformtation going
# from voxel coordinates in the referenced 3D voxel matrix, to xyz
# coordinates (axes L->R, P->A, I->S).  IF (0 based) value [3, 3] from
# this matrix is 0, this means the matrix is not recorded.
header_2_dtd = [
    ('id_string', 'S6'),
    ('dim', 'h', 3),
    ('voxel_size', 'f4', 3),
    ('origin', 'f4', 3),
    ('n_scalars', 'h'),
    ('scalar_name', 'S20', 10),
    ('n_properties', 'h'),
    ('property_name', 'S20', 10),
    ('vox_to_ras', 'f4', (4,4)), # new field for version 2
    ('reserved', 'S444'),
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

# Full header numpy dtypes
header_1_dtype = np.dtype(header_1_dtd)
header_2_dtype = np.dtype(header_2_dtd)

# affine to go from DICOM LPS to MNI RAS space
DPCS_TO_TAL = np.diag([-1, -1, 1, 1])


class HeaderError(Exception):
    pass


class DataError(Exception):
    pass


def read(fileobj):
    ''' Read trackvis file, return header, streamlines

    Parameters
    ----------
    fileobj : string or file-like object
       If string, a filename; otherwise an open file-like object
       pointing to trackvis file (and ready to read from the beginning
       of the trackvis header data)

    Returns
    -------
    streamlines : sequence
       sequence of 3 element sequences with elements:

       #. points : ndarray shape (N,3)
          where N is the number of points
       #. scalars : None or ndarray shape (N, M)
          where M is the number of scalars per point
       #. properties : None or ndarray shape (P,)
          where P is the number of properties
    hdr : structured array
       structured array with trackvis header fields

    Notes
    -----
    The endianness of the input data can be deduced from the endianness
    of the returned `hdr` or `streamlines`
    '''
    fileobj = allopen(fileobj, mode='rb')
    hdr_str = fileobj.read(header_2_dtype.itemsize)
    # try defaulting to version 2 format
    hdr = np.ndarray(shape=(),
                     dtype=header_2_dtype,
                     buffer=hdr_str)
    if str(hdr['id_string'])[:5] != 'TRACK':
        raise HeaderError('Expecting TRACK as first '
                          '5 characters of id_string')
    version = hdr['version']
    if version not in (1, 2):
        raise HeaderError('Reader only supports versions 1 and 2')
    if version == 1:
        hdr = np.ndarray(shape=(),
                         dtype=header_1_dtype,
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
        streamlines.append((xyz, scalars, ps))
        n_streams += 1
        # deliberately misses case where stream_count is 0
        if n_streams == stream_count:
            break
    return streamlines, hdr


def write(fileobj, streamlines,  hdr_mapping=None, endianness=None):
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
    streamlines : sequence
       sequence of 3 element sequences with elements:

       #. points : ndarray shape (N,3)
          where N is the number of points
       #. scalars : None or ndarray shape (N, M)
          where M is the number of scalars per point
       #. properties : None or ndarray shape (P,)
          where P is the number of properties

    hdr_mapping : None, ndarray or mapping, optional
       Information for filling header fields.  Can be something
       dict-like (implementing ``items``) or a structured numpy array
    endianness : {None, '<', '>'}, optional
       Endianness of file to be written.  '<' is little-endian, '>' is
       big-endian.  None (the default) is to use the endianness of the
       `streamlines` data.

    Returns
    -------
    None
    '''
    # First work out guessed endianness from input data
    if endianness is None:
        endianness = _endian_from_streamlines(streamlines)
    # fill in a new header from mapping-like
    hdr = _hdr_from_mapping(None, hdr_mapping, endianness)
    # put calculated data into header
    stream_count = len(streamlines)
    hdr['n_count'] = stream_count
    if stream_count:
        pts, scalars, props = streamlines[0]
        # calculate number of scalars
        if scalars:
            n_s = scalars.shape[1]
        else:
            n_s = 0
        hdr['n_scalars'] = n_s
        # calculate number of properties
        if props:
            n_p = props.size
            hdr['n_properties'] = n_p
        else:
            n_p = 0
    # write header
    fileobj = allopen(fileobj, mode='wb')
    fileobj.write(hdr.tostring())
    if not stream_count:
        return
    f4dt = np.dtype(endianness + 'f4')
    i_fmt = endianness + 'i'
    for pts, scalars, props in streamlines:
        n_pts, n_coords = pts.shape
        if n_coords != 3:
            raise ValueError('pts should have 3 columns')
        fileobj.write(struct.pack(i_fmt, n_pts))
        # This call ensures that the data are 32-bit floats, and that
        # the endianness is OK.
        if pts.dtype != f4dt:
            pts = pts.astype(f4dt)
        if n_s:
            if scalars.shape != (n_pts, n_s):
                raise ValueError('Scalars should be shape (%s, %s)'
                                 % (n_pts, n_s))
            if scalars.dtype != f4dt:
                scalars = scalars.astype(f4dt)
                pts = np.c_[pts, scalars]
        fileobj.write(pts.tostring())
        if n_p:
            if props.size != n_p:
                raise ValueError('Properties should be size %s' % n_p)
            if props.dtype != f4dt:
                props = props.astype(f4dt)
            fileobj.write(props.tostring())


def _endian_from_streamlines(streamlines):
    if len(streamlines) == 0:
        return native_code
    endian = streamlines[0][0].dtype.byteorder
    return endian_codes[endian]


def _hdr_from_mapping(hdr=None, mapping=None, endianness=native_code):
    ''' Fill `hdr` from mapping `mapping`, with given endianness '''
    if hdr is None:
        # passed a valid mapping as header?  Copy and return
        if isinstance(mapping, np.ndarray):
            test_dtype = mapping.dtype.newbyteorder('=')
            if test_dtype in (header_1_dtype, header_2_dtype):
                return mapping.copy()
        # otherwise make a new empty header.   If no version specified,
        # go for default (2)
        if mapping is None:
            version = 2
        else:
            version =  mapping.get('version', 2)
        hdr = empty_header(endianness, version)
    if mapping is None:
        return hdr
    if isinstance(mapping, np.ndarray):
        mapping = rec2dict(mapping)
    for key, value in mapping.items():
        hdr[key] = value
    # check header values
    if str(hdr['id_string'])[:5] != 'TRACK':
        raise HeaderError('Expecting TRACK as first '
                          '5 characaters of id_string')
    if hdr['version'] not in (1, 2):
        raise HeaderError('Reader only supports version 1')
    if hdr['hdr_size'] != 1000:
        raise HeaderError('hdr_size should be 1000')
    return hdr


def empty_header(endianness=None, version=2):
    ''' Empty trackvis header
    
    Parameters
    ----------
    endianness : {'<','>'}, optional
       Endianness of empty header to return. Default is native endian.
    version : int, optional
       Header version.  1 or 2.  Default is 2
      
    Returns
    -------
    hdr : structured array
       structured array containing empty trackvis header

    Examples
    --------
    >>> hdr = empty_header()
    >>> print hdr['version']
    2
    >>> print hdr['id_string']
    TRACK
    >>> endian_codes[hdr['version'].dtype.byteorder] == native_code
    True
    >>> hdr = empty_header(swapped_code)
    >>> endian_codes[hdr['version'].dtype.byteorder] == swapped_code
    True
    >>> hdr = empty_header(version=1)
    >>> print hdr['version']
    1

    Notes
    -----
    The trackviz header can store enough information to give an affine
    mapping between voxel and world space.  Often this information is
    missing.  We make no attempt to fill it with sensible defaults on
    the basis that, if the information is missing, it is better to be
    explicit.
    '''
    if version == 1:
        dt = header_1_dtype
    elif version == 2:
        dt = header_2_dtype
    else:
        raise HeaderError('Header version should be 1 or 2')
    if endianness:
        dt = dt.newbyteorder(endianness)
    hdr = np.zeros((), dtype=dt)
    hdr['id_string'] = 'TRACK'
    hdr['version'] = version
    hdr['hdr_size'] = 1000
    return hdr


def aff_from_hdr(trk_hdr):
    ''' Return voxel to mm affine from trackvis header

    Affine is mapping from voxel space to Nifti (RAS) output coordinate
    system convention; x: Left -> Right, y: Posterior -> Anterior, z:
    Inferior -> Superior.
    
    Parameters
    ----------
    trk_hdr : mapping
       Mapping with trackvis header keys ``version``,
       ``image_orientation_patient``, ``voxel_size`` and ``origin``.  If
       ``version == 2``, we also expect ``vox_to_ras``.

    Returns
    -------
    aff : (4,4) array
       affine giving mapping from voxel coordinates (affine applied on
       the left to points on the right) to millimeter coordinates in the
       RAS coordinate system
    '''
    if trk_hdr['version'] == 2:
        aff = trk_hdr['vox_to_ras']
        if aff[3,3] != 0:
            return aff
    aff = np.eye(4)
    iop = trk_hdr['image_orientation_patient'].reshape(2,3).T
    R = np.c_[iop, np.cross(*iop.T)]
    vox = trk_hdr['voxel_size']
    aff[:3,:3] = R * vox
    aff[:3,3] = trk_hdr['origin']
    return np.dot(DPCS_TO_TAL, aff)


def aff_to_hdr(affine, trk_hdr):
    ''' Set affine `affine` into trackvix header `trk_hdr`

    Affine is mapping from voxel space to Nifti RAS) output coordinate
    system convention; x: Left -> Right, y: Posterior -> Anterior, z:
    Inferior -> Superior.
    
    Parameters
    ----------
    affine : (4,4) array-like
       Affine voxel to mm transformation
    trk_hdr : mapping
       Mapping implementing __setitem__

    Returns
    -------
    None
    '''
    try:
        version = trk_hdr['version']
    except KeyError:
        version = 2
    if version == 2:
        trk_hdr['vox_to_ras'] = affine
    # RAS to DPCS output
    affine = np.dot(DPCS_TO_TAL, affine)
    trans = affine[:3, 3]
    # Get zooms
    RZS = affine[:3, :3]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    RS = RZS / zooms
    # adjust zooms to make RS correspond (below) to a true rotation
    # matrix.  We need to set the sign of one of the zooms to deal with
    # this.
    if npl.det(RS) < 0:
        zooms[0] *= -1
        RS[:,0] *= -1
    # retrieve rotation matrix from RS with polar decomposition.
    # Discard shears because we cannot store them.
    P, S, Qs = npl.svd(RS)
    R = np.dot(P, Qs)
    # it's a rotation matrix
    assert np.allclose(np.dot(R, R.T), np.eye(3))
    # set into header
    trk_hdr['origin'] = trans
    trk_hdr['voxel_size'] = zooms
    trk_hdr['image_orientation_patient'] = R[:,0:2].T.ravel()


class TrackvisFile(object):
    ''' Convenience class to encapsulate trackviz file information '''
    def __init__(self,
                 streamlines,
                 mapping=None,
                 endianness=None,
                 filename=None):
        self.streamlines = streamlines
        if endianness is None:
            endianness = _endian_from_streamlines(streamlines)
        self.header = _hdr_from_mapping(None, mapping, endianness)
        self.endianness = endianness
        self.filename = filename

    @classmethod
    def from_file(klass, file_like):
        streamlines, header = read(file_like)
        filename = (file_like if isinstance(file_like, basestring)
                    else None)
        return klass(streamlines, header, None, filename)

    def to_file(self, file_like):
        write(file_like, self.streamlines, self.header, self.endianness)
        self.filename = (file_like if isinstance(file_like, basestring)
                         else None)

    def get_affine(self):
        # use method becase set may involve removing shears from affine
        return aff_from_hdr(self.header)

    def set_affine(self, affine):
        # use method becase set may involve removing shears from affine
        return aff_to_hdr(affine, self.header)
