from bisect import bisect
from collections import OrderedDict
from copy import deepcopy
import enum
from itertools import product
import logging

from nibabel.affines import apply_affine
from nibabel.streamlines.tractogram import (Tractogram,
                                            PerArraySequenceDict,
                                            PerArrayDict)
import numpy as np

from dipy.io.dpy import Streamlines
from dipy.io.utils import (get_reference_info,
                           is_reference_info_valid,
                           is_header_compatible)

logger = logging.getLogger('StatefulTractogram')
logger.setLevel(level=logging.INFO)


def set_sft_logger_level(log_level):
    """ Change the logger of the StatefulTractogram
    to one on the following: DEBUG, INFO, WARNING, CRITICAL, ERROR

    Parameters
    ----------
    log_level : str
        Log level for the StatefulTractogram only
    """
    logger.setLevel(level=log_level)


class Space(enum.Enum):
    """ Enum to simplify future change to convention """
    VOX = 'vox'
    VOXMM = 'voxmm'
    RASMM = 'rasmm'


class Origin(enum.Enum):
    """ Enum to simplify future change to convention """
    NIFTI = 'center'
    TRACKVIS = 'corner'


class StatefulTractogram:
    """ Class for stateful representation of collections of streamlines
    Object designed to be identical no matter the file format
    (trk, tck, vtk, fib, dpy). Facilitate transformation between space and
    data manipulation for each streamline / point.
    """

    def __init__(self, streamlines, reference, space,
                 origin=Origin.NIFTI,
                 data_per_point=None, data_per_streamline=None):
        """ Create a strict, state-aware, robust tractogram

        Parameters
        ----------
        streamlines : list or ArraySequence
            Streamlines of the tractogram
        reference : Nifti or Trk filename, Nifti1Image or TrkFile,
            Nifti1Header, trk.header (dict) or another Stateful Tractogram
            Reference that provides the spatial attributes.
            Typically a nifti-related object from the native diffusion used for
            streamlines generation
        space : Enum (dipy.io.stateful_tractogram.Space)
            Current space in which the streamlines are (vox, voxmm or rasmm)
            After tracking the space is VOX, after loading with nibabel
            the space is RASMM
        origin : Enum (dipy.io.stateful_tractogram.Origin), optional
            Current origin in which the streamlines are (center or corner)
            After loading with nibabel the origin is CENTER
        data_per_point : dict, optional
            Dictionary in which each key has X items, each items has Y_i items
            X being the number of streamlines
            Y_i being the number of points on streamlines #i
        data_per_streamline : dict, optional
            Dictionary in which each key has X items
            X being the number of streamlines

        Notes
        -----
        Very important to respect the convention, verify that streamlines
        match the reference and are effectively in the right space.

        Any change to the number of streamlines, data_per_point or
        data_per_streamline requires particular verification.

        In a case of manipulation not allowed by this object, use Nibabel
        directly and be careful.
        """
        if data_per_point is None:
            data_per_point = {}

        if data_per_streamline is None:
            data_per_streamline = {}

        if isinstance(streamlines, Streamlines):
            streamlines = streamlines.copy()

        self._tractogram = Tractogram(streamlines,
                                      data_per_point=data_per_point,
                                      data_per_streamline=data_per_streamline)

        if isinstance(reference, type(self)):
            logger.warning('Using a StatefulTractogram as reference, this '
                           'will copy only the space_attributes, not '
                           'the state. The variables space and origin '
                           'must be specified separately.')
            logger.warning('To copy the state from another StatefulTractogram '
                           'you may want to use the function from_sft '
                           '(static function of the StatefulTractogram).')

        if isinstance(reference, tuple) and len(reference) == 4:
            if is_reference_info_valid(*reference):
                space_attributes = reference
            else:
                raise TypeError('The provided space attributes are not '
                                'considered valid, please correct before '
                                'using them with StatefulTractogram.')
        else:
            space_attributes = get_reference_info(reference)
            if space_attributes is None:
                raise TypeError('Reference MUST be one of the following:\n'
                                'Nifti or Trk filename, Nifti1Image or '
                                'TrkFile, Nifti1Header or trk.header (dict).')

        (self._affine, self._dimensions,
         self._voxel_sizes, self._voxel_order) = space_attributes
        self._inv_affine = np.linalg.inv(self._affine).astype(np.float32)

        if space not in Space:
            raise ValueError('Space MUST be from Space enum, e.g Space.VOX.')
        self._space = space

        if origin not in Origin:
            raise ValueError('Origin MUST be from Origin enum, '
                             'e.g Origin.NIFTI.')
        self._origin = origin

        logger.debug(self)

    @staticmethod
    def are_compatible(sft_1, sft_2):
        """ Compatibility verification of two StatefulTractogram to ensure space,
        origin, data_per_point and data_per_streamline consistency """

        are_sft_compatible = True
        if not is_header_compatible(sft_1, sft_2):
            logger.warning('Inconsistent spatial attributes between both sft.')
            are_sft_compatible = False

        if sft_1.space != sft_2.space:
            logger.warning('Inconsistent space between both sft.')
            are_sft_compatible = False
        if sft_1.origin != sft_2.origin:
            logger.warning('Inconsistent origin between both sft.')
            are_sft_compatible = False

        if sft_1.get_data_per_point_keys() != sft_2.get_data_per_point_keys():
            logger.warning(
                'Inconsistent data_per_point between both sft.')
            are_sft_compatible = False
        if sft_1.get_data_per_streamline_keys() != \
                sft_2.get_data_per_streamline_keys():
            logger.warning(
                'Inconsistent data_per_streamline between both sft.')
            are_sft_compatible = False

        return are_sft_compatible

    @staticmethod
    def from_sft(streamlines, sft,
                 data_per_point=None,
                 data_per_streamline=None):
        """ Create an instance of `StatefulTractogram` from another instance
        of `StatefulTractogram`.

        Parameters
        ----------
        streamlines : list or ArraySequence
            Streamlines of the tractogram
        sft : StatefulTractogram,
            The other StatefulTractogram to copy the space_attribute AND
            state from.
        data_per_point : dict, optional
            Dictionary in which each key has X items, each items has Y_i items
            X being the number of streamlines
            Y_i being the number of points on streamlines #i
        data_per_streamline : dict, optional
            Dictionary in which each key has X items
            X being the number of streamlines
        -----
        """
        new_sft = StatefulTractogram(streamlines,
                                     sft.space_attributes,
                                     sft.space,
                                     origin=sft.origin,
                                     data_per_point=data_per_point,
                                     data_per_streamline=data_per_streamline)
        new_sft.dtype_dict = sft.dtype_dict
        return new_sft

    def __str__(self):
        """ Generate the string for printing """
        text = 'Affine: \n{}'.format(
            np.array2string(self._affine,
                            formatter={'float_kind': lambda x: "%.6f" % x}))
        text += '\ndimensions: {}'.format(
            np.array2string(self._dimensions))
        text += '\nvoxel_sizes: {}'.format(
            np.array2string(self._voxel_sizes,
                            formatter={'float_kind': lambda x: "%.2f" % x}))
        text += '\nvoxel_order: {}'.format(self._voxel_order)

        text += '\nstreamline_count: {}'.format(self._get_streamline_count())
        text += '\npoint_count: {}'.format(self._get_point_count())
        text += '\ndata_per_streamline keys: {}'.format(
            self.get_data_per_streamline_keys())
        text += '\ndata_per_point keys: {}'.format(
            self.get_data_per_point_keys())

        return text

    def __len__(self):
        """ Define the length of the object """
        return self._get_streamline_count()

    def __getitem__(self, key):
        """ Slice all data in a consistent way """
        if isinstance(key, int):
            key = [key]

        return self.from_sft(self.streamlines[key], self,
                             data_per_point=self.data_per_point[key],
                             data_per_streamline=self.data_per_streamline[key])

    def __eq__(self, other):
        """ Robust StatefulTractogram equality test """
        if not self.are_compatible(self, other):
            return False

        streamlines_equal = np.allclose(self.streamlines.get_data(),
                                        other.streamlines.get_data(),
                                        rtol=1e-3)
        if not streamlines_equal:
            return False

        dpp_equal = True
        for key in self.data_per_point:
            dpp_equal = dpp_equal and np.allclose(
                self.data_per_point[key].get_data(),
                other.data_per_point[key].get_data(),
                rtol=1e-3)
        if not dpp_equal:
            return False

        dps_equal = True
        for key in self.data_per_streamline:
            dps_equal = dps_equal and np.allclose(
                self.data_per_streamline[key],
                other.data_per_streamline[key],
                rtol=1e-3)
        if not dps_equal:
            return False

        return True

    def __ne__(self, other):
        """ Robust StatefulTractogram equality test (NOT) """
        return not self == other

    def __add__(self, other_sft):
        """ Addition of two sft with attributes consistency checks """
        if not self.are_compatible(self, other_sft):
            logger.debug(self)
            logger.debug(other_sft)
            raise ValueError('Inconsistent StatefulTractogram.\n'
                             'Make sure Space, Origin are the same and that '
                             'data_per_point and data_per_streamline keys are '
                             'the same.')

        streamlines = self.streamlines.copy()
        streamlines.extend(other_sft.streamlines)

        data_per_point = deepcopy(self.data_per_point)
        data_per_point.extend(other_sft.data_per_point)

        data_per_streamline = deepcopy(self.data_per_streamline)
        data_per_streamline.extend(other_sft.data_per_streamline)

        return self.from_sft(streamlines, self,
                             data_per_point=data_per_point,
                             data_per_streamline=data_per_streamline)

    def __iadd__(self, other):
        self.value = self + other
        return self.value

    @property
    def dtype_dict(self):
        """ Getter for dtype_dict """

        dtype_dict = {'positions': self.streamlines._data.dtype,
                      'offsets': self.streamlines._offsets.dtype}
        if self.data_per_point is not None:
            dtype_dict['dpp'] = {}
            for key in self.data_per_point.keys():
                if key in self.data_per_point:
                    dtype_dict['dpp'][key] = self.data_per_point[key]._data.dtype
        if self.data_per_streamline is not None:
            dtype_dict['dps'] = {}
            for key in self.data_per_streamline.keys():
                if key in self.data_per_streamline:
                    dtype_dict['dps'][key] = self.data_per_streamline[key].dtype
        return OrderedDict(dtype_dict)

    @property
    def space_attributes(self):
        """ Getter for spatial attribute """
        return self._affine, self._dimensions, self._voxel_sizes, \
            self._voxel_order

    @property
    def space(self):
        """ Getter for the current space """
        return self._space

    @property
    def affine(self):
        """ Getter for the reference affine """
        return self._affine

    @property
    def dimensions(self):
        """ Getter for the reference dimensions """
        return self._dimensions

    @property
    def voxel_sizes(self):
        """ Getter for the reference voxel sizes """
        return self._voxel_sizes

    @property
    def voxel_order(self):
        """ Getter for the reference voxel order """
        return self._voxel_order

    @property
    def origin(self):
        """ Getter for origin standard """
        return self._origin

    @property
    def streamlines(self):
        """ Partially safe getter for streamlines """
        return self._tractogram.streamlines

    @dtype_dict.setter
    def dtype_dict(self, dtype_dict):
        """ Modify dtype_dict.

        Parameters
        ----------
        dtype_dict : dict
            Dictionary containing the desired datatype for positions, offsets
            and all dpp and dps keys. (To use with TRX file format):
        """
        if 'offsets' in dtype_dict:
            self.streamlines._offsets = self.streamlines._offsets.astype(
                dtype_dict['offsets'])
        if 'positions' in dtype_dict:
            self.streamlines._data = self.streamlines._data.astype(
                dtype_dict['positions'])

        if 'dpp' not in dtype_dict:
            dtype_dict['dpp'] = {}
        if 'dps' not in dtype_dict:
            dtype_dict['dps'] = {}

        for key in self.data_per_point:
            if key in dtype_dict['dpp']:
                dtype_to_use = dtype_dict['dpp'][key]
                self.data_per_point[key]._data = \
                    self.data_per_point[key]._data.astype(dtype_to_use)

        for key in self.data_per_streamline:
            if key in dtype_dict['dps']:
                dtype_to_use = dtype_dict['dps'][key]
                self.data_per_streamline[key] = \
                    self.data_per_streamline[key].astype(dtype_to_use)

    def get_streamlines_copy(self):
        """ Safe getter for streamlines (for slicing) """
        return self._tractogram.streamlines.copy()

    @streamlines.setter
    def streamlines(self, streamlines):
        """ Modify streamlines. Creating a new object would be less risky.

        Parameters
        ----------
        streamlines : list or ArraySequence (list and deepcopy recommended)
            Streamlines of the tractogram
        """
        if isinstance(streamlines, Streamlines):
            streamlines = streamlines.copy()
        self._tractogram._streamlines = Streamlines(streamlines)
        self.data_per_point = self.data_per_point
        self.data_per_streamline = self.data_per_streamline
        logger.warning('Streamlines has been modified.')

    @property
    def data_per_point(self):
        """ Getter for data_per_point """
        return self._tractogram.data_per_point

    @data_per_point.setter
    def data_per_point(self, data):
        """ Modify point data . Creating a new object would be less risky.

        Parameters
        ----------
        data : dict
            Dictionary in which each key has X items, each items has Y_i items
            X being the number of streamlines
            Y_i being the number of points on streamlines #i
        """
        self._tractogram.data_per_point = data
        logger.warning('Data_per_point has been modified.')

    @property
    def data_per_streamline(self):
        """ Getter for data_per_streamline """
        return self._tractogram.data_per_streamline

    @data_per_streamline.setter
    def data_per_streamline(self, data):
        """ Modify point data . Creating a new object would be less risky.

        Parameters
        ----------
        data : dict
            Dictionary in which each key has X items, each items has Y_i items
            X being the number of streamlines
        """
        self._tractogram.data_per_streamline = data
        logger.warning('Data_per_streamline has been modified.')

    def get_data_per_point_keys(self):
        """ Return a list of the data_per_point attribute names """
        return list(set(self.data_per_point.keys()))

    def get_data_per_streamline_keys(self):
        """ Return a list of the data_per_streamline attribute names """
        return list(set(self.data_per_streamline.keys()))

    def to_vox(self):
        """ Safe function to transform streamlines and update state """
        if self._space == Space.VOXMM:
            self._voxmm_to_vox()
        elif self._space == Space.RASMM:
            self._rasmm_to_vox()

    def to_voxmm(self):
        """ Safe function to transform streamlines and update state """
        if self._space == Space.VOX:
            self._vox_to_voxmm()
        elif self._space == Space.RASMM:
            self._rasmm_to_voxmm()

    def to_rasmm(self):
        """ Safe function to transform streamlines and update state """
        if self._space == Space.VOX:
            self._vox_to_rasmm()
        elif self._space == Space.VOXMM:
            self._voxmm_to_rasmm()

    def to_space(self, target_space):
        """ Safe function to transform streamlines to a particular space using
        an enum and update state """
        if target_space == Space.VOX:
            self.to_vox()
        elif target_space == Space.VOXMM:
            self.to_voxmm()
        elif target_space == Space.RASMM:
            self.to_rasmm()
        else:
            logger.error('Unsupported target space, please use Enum in '
                         'dipy.io.stateful_tractogram.')

    def to_origin(self, target_origin):
        """ Safe function to change streamlines to a particular origin standard
        False means NIFTI (center) and True means TrackVis (corner) """
        if target_origin == Origin.NIFTI:
            self.to_center()
        elif target_origin == Origin.TRACKVIS:
            self.to_corner()
        else:
            logger.error('Unsupported origin standard, please use Enum in '
                         'dipy.io.stateful_tractogram.')

    def to_center(self):
        """ Safe function to shift streamlines so the center of voxel is
        the origin """
        if self._origin == Origin.TRACKVIS:
            self._shift_voxel_origin()

    def to_corner(self):
        """ Safe function to shift streamlines so the corner of voxel is
        the origin """
        if self._origin == Origin.NIFTI:
            self._shift_voxel_origin()

    def compute_bounding_box(self):
        """ Compute the bounding box of the streamlines in their current state

        Returns
        -------
        output : ndarray
            8 corners of the XYZ aligned box, all zeros if no streamlines
        """
        if self._tractogram.streamlines._data.size > 0:
            bbox_min = np.min(self._tractogram.streamlines._data, axis=0)
            bbox_max = np.max(self._tractogram.streamlines._data, axis=0)
            return np.asarray(list(product(*zip(bbox_min, bbox_max))))

        return np.zeros((8, 3))

    def is_bbox_in_vox_valid(self):
        """ Verify that the bounding box is valid in voxel space.
        Negative coordinates or coordinates above the volume dimensions
        are considered invalid in voxel space.

        Returns
        -------
        output : bool
            Are the streamlines within the volume of the associated reference
        """
        if not self.streamlines:
            return True

        old_space = deepcopy(self.space)
        old_origin = deepcopy(self.origin)

        # Do to rotation, equivalent of a OBB must be done
        self.to_vox()
        self.to_corner()
        bbox_corners = deepcopy(self.compute_bounding_box())

        is_valid = True
        if np.any(bbox_corners < 0):
            logger.error('Voxel space values lower than 0.0.')
            logger.debug(bbox_corners)
            is_valid = False

        if np.any(bbox_corners[:, 0] > self._dimensions[0]) or \
                np.any(bbox_corners[:, 1] > self._dimensions[1]) or \
                np.any(bbox_corners[:, 2] > self._dimensions[2]):
            logger.error('Voxel space values higher than dimensions.')
            logger.debug(bbox_corners)
            is_valid = False

        self.to_space(old_space)
        self.to_origin(old_origin)

        return is_valid

    def remove_invalid_streamlines(self, epsilon=1e-3):
        """ Remove streamlines with invalid coordinates from the object.
        Will also remove the data_per_point and data_per_streamline.
        Invalid coordinates are any X,Y,Z values above the reference
        dimensions or below zero

        Parameters
        ----------
        epsilon : float (optional)
            Epsilon value for the bounding box verification.
            Default is 1e-6.

        Returns
        -------
        output : tuple
            Tuple of two list, indices_to_remove, indices_to_keep
        """
        if not self.streamlines:
            return

        old_space = deepcopy(self.space)
        old_origin = deepcopy(self.origin)

        self.to_vox()
        self.to_corner()

        min_condition = np.min(self._tractogram.streamlines._data,
                               axis=1) < epsilon
        max_condition = np.any(self._tractogram.streamlines._data >
                               self._dimensions-epsilon, axis=1)
        ic_offsets_indices = np.where(np.logical_or(min_condition,
                                                    max_condition))[0]

        indices_to_remove = []
        for i in ic_offsets_indices:
            indices_to_remove.append(bisect(
                self._tractogram.streamlines._offsets, i) - 1)

        indices_to_remove = sorted(set(indices_to_remove))

        indices_to_keep = list(
            np.setdiff1d(np.arange(len(self._tractogram)),
                         np.array(indices_to_remove)).astype(int))

        tmp_streamlines = self.streamlines[indices_to_keep]
        tmp_dpp = self._tractogram.data_per_point[indices_to_keep]
        tmp_dps = self._tractogram.data_per_streamline[indices_to_keep]

        ori_dtype = self._tractogram.streamlines._data.dtype
        tmp_streamlines = tmp_streamlines.copy()
        tmp_streamlines._data = tmp_streamlines._data.astype(ori_dtype)
        self._tractogram = Tractogram(tmp_streamlines,
                                      data_per_point=tmp_dpp,
                                      data_per_streamline=tmp_dps,
                                      affine_to_rasmm=np.eye(4))

        self.to_space(old_space)
        self.to_origin(old_origin)

        return indices_to_remove, indices_to_keep

    def _get_streamline_count(self):
        """ Safe getter for the number of streamlines """
        return len(self._tractogram)

    def _get_point_count(self):
        """ Safe getter for the number of streamlines """
        return self._tractogram.streamlines.total_nb_rows

    def _vox_to_voxmm(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.VOX:
            if self._tractogram.streamlines._data.size > 0:
                self._tractogram.streamlines._data *= np.asarray(
                    self._voxel_sizes)
            self._space = Space.VOXMM
            logger.debug('Moved streamlines from vox to voxmm.')
        else:
            logger.warning('Wrong initial space for this function.')

    def _voxmm_to_vox(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.VOXMM:
            if self._tractogram.streamlines._data.size > 0:
                self._tractogram.streamlines._data /= np.asarray(
                    self._voxel_sizes)
            self._space = Space.VOX
            logger.debug('Moved streamlines from voxmm to vox.')
        else:
            logger.warning('Wrong initial space for this function.')

    def _vox_to_rasmm(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.VOX:
            if self._tractogram.streamlines._data.size > 0:
                self._tractogram.apply_affine(self._affine)
            self._space = Space.RASMM
            logger.debug('Moved streamlines from vox to rasmm.')
        else:
            logger.warning('Wrong initial space for this function.')

    def _rasmm_to_vox(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.RASMM:
            if self._tractogram.streamlines._data.size > 0:
                self._tractogram.apply_affine(self._inv_affine)
            self._space = Space.VOX
            logger.debug('Moved streamlines from rasmm to vox.')
        else:
            logger.warning('Wrong initial space for this function.')

    def _voxmm_to_rasmm(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.VOXMM:
            if self._tractogram.streamlines._data.size > 0:
                self._tractogram.streamlines._data /= np.asarray(
                    self._voxel_sizes)
                self._tractogram.apply_affine(self._affine)
            self._space = Space.RASMM
            logger.debug('Moved streamlines from voxmm to rasmm.')
        else:
            logger.warning('Wrong initial space for this function.')

    def _rasmm_to_voxmm(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.RASMM:
            if self._tractogram.streamlines._data.size > 0:
                self._tractogram.apply_affine(self._inv_affine)
                self._tractogram.streamlines._data *= np.asarray(
                    self._voxel_sizes)
            self._space = Space.VOXMM
            logger.debug('Moved streamlines from rasmm to voxmm.')
        else:
            logger.warning('Wrong initial space for this function.')

    def _shift_voxel_origin(self):
        """ Unsafe function to switch the origin from center to corner
        and vice versa """
        if self.streamlines:
            shift = np.asarray([0.5, 0.5, 0.5])
            if self._space == Space.VOXMM:
                shift = shift * self._voxel_sizes
            elif self._space == Space.RASMM:
                tmp_affine = np.eye(4)
                tmp_affine[0:3, 0:3] = self._affine[0:3, 0:3]
                shift = apply_affine(tmp_affine, shift)
            if self._origin == Origin.TRACKVIS:
                shift *= -1

            self._tractogram.streamlines._data += shift

        if self._origin == Origin.NIFTI:
            logger.debug('Origin moved to the corner of voxel.')
            self._origin = Origin.TRACKVIS
        else:
            logger.debug('Origin moved to the center of voxel.')
            self._origin = Origin.NIFTI


def _is_data_per_point_valid(streamlines, data):
    """ Verify that the number of item in data is X and that each of these
        items has Y_i items.

        X being the number of streamlines
        Y_i being the number of points on streamlines #i

    Parameters
    ----------
    streamlines : list or ArraySequence
        Streamlines of the tractogram
    data : dict
        Contains the organized point's metadata (hopefully)
    Returns
    -------
    output : bool
        Does all the streamlines and metadata attribute match
    """
    if not isinstance(data, (dict, PerArraySequenceDict)):
        logger.error('data_per_point MUST be a dictionary.')
        return False
    elif data == {}:
        return True

    total_point = 0
    total_streamline = 0
    for i in streamlines:
        total_streamline += 1
        total_point += len(i)

    for key in data.keys():
        total_point_entries = 0
        if not len(data[key]) == total_streamline:
            logger.error('Missing entry for streamlines points data, '
                         'inconsistent number of streamlines.')
            return False

        for values in data[key]:
            total_point_entries += len(values)

        if total_point_entries != total_point:
            logger.error('Missing entry for streamlines points data, '
                         'inconsistent number of points per streamlines.')
            return False

    return True


def _is_data_per_streamline_valid(streamlines, data):
    """ Verify that the number of item in data is X

        X being the number of streamlines

    Parameters
    ----------
    streamlines : list or ArraySequence
        Streamlines of the tractogram
    data : dict
        Contains the organized streamline's metadata (hopefully)
    Returns
    -------
    output : bool
        Does all the streamlines and metadata attribute match
    """
    if not isinstance(data, (dict, PerArrayDict)):
        logger.error('data_per_point MUST be a dictionary.')
        return False
    elif data == {}:
        return True

    total_streamline = 0
    for _ in streamlines:
        total_streamline += 1

    for key in data.keys():
        if not len(data[key]) == total_streamline:
            logger.error('Missing entry for streamlines points data, '
                         'inconsistent number of streamlines.')
            return False

    return True
