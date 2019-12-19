from bisect import bisect
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
from dipy.io.utils import get_reference_info, is_reference_info_valid


logger = logging.getLogger('StatefulTractogram')


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


class StatefulTractogram(object):
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
            logging.warning('Using a StatefulTractogram as reference, this '
                            'will copy only the space_attributes, not '
                            'the state. The variables space and origin '
                            'must be specified separately.')
            logging.warning('To copy the state from another StatefulTractogram'
                            'you may want to use the function from_sft'
                            '(static function of the StatefulTractogram)')

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
        self._inv_affine = np.linalg.inv(self._affine)

        if space not in Space:
            raise ValueError('Space MUST be from Space enum, e.g Space.VOX')
        self._space = space

        if origin not in Origin:
            raise ValueError('Origin MUST be from Origin enum, '
                             'e.g Origin.NIFTI')
        self._origin = origin
        logger.debug(self)

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
        sft : StatefulTractgram,
            The other StatefulTractgram to copy the space_attribute AND
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
            self.data_per_point.keys())
        text += '\ndata_per_point keys: {}'.format(
            self.data_per_streamline.keys())

        return text

    def __len__(self):
        """ Define the length of the object """
        return self._get_streamline_count()

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

    def get_streamlines_copy(self):
        """ Safe getter for streamlines (for slicing) """
        return self._tractogram.streamlines.copy()

    @streamlines.setter
    def streamlines(self, streamlines):
        """ Modify streamlines. Creating a new object would be less risky.

        Parameters
        ----------
        streamlines : list or ArraySequence (list and deepcopy recommanded)
            Streamlines of the tractogram
        """
        if isinstance(streamlines, Streamlines):
            streamlines = streamlines.copy()
        self._tractogram._streamlines = Streamlines(streamlines)
        self.data_per_point = self.data_per_point
        self.data_per_streamline = self.data_per_streamline
        logger.warning('Streamlines has been modified')

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
        logger.warning('Data_per_point has been modified')

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
        logger.warning('Data_per_streamline has been modified')

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
                         'dipy.io.stateful_tractogram')

    def to_origin(self, target_origin):
        """ Safe function to change streamlines to a particular origin standard
        False means NIFTI (center) and True means TrackVis (corner) """
        if target_origin == Origin.NIFTI:
            self.to_center()
        elif target_origin == Origin.TRACKVIS:
            self.to_corner()
        else:
            logger.error('Unsupported origin standard, please use Enum in '
                         'dipy.io.stateful_tractogram')

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
            logger.error('Voxel space values lower than 0.0')
            logger.debug(bbox_corners)
            is_valid = False

        if np.any(bbox_corners[:, 0] > self._dimensions[0]) or \
                np.any(bbox_corners[:, 1] > self._dimensions[1]) or \
                np.any(bbox_corners[:, 2] > self._dimensions[2]):
            logger.error('Voxel space values higher than dimensions')
            logger.debug(bbox_corners)
            is_valid = False

        self.to_space(old_space)
        self.to_origin(old_origin)

        return is_valid

    def remove_invalid_streamlines(self):
        """ Remove streamlines with invalid coordinates from the object.
        Will also remove the data_per_point and data_per_streamline.
        Invalid coordinates are any X,Y,Z values above the reference
        dimensions or below zero
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
                               axis=1) < 0.0
        max_condition = np.any(self._tractogram.streamlines._data >
                               self._dimensions, axis=1)
        ic_offsets_indices = np.where(np.logical_or(min_condition,
                                                    max_condition))[0]

        indices_to_remove = []
        for i in ic_offsets_indices:
            indices_to_remove.append(bisect(
                self._tractogram.streamlines._offsets, i) - 1)

        indices_to_keep = np.setdiff1d(np.arange(len(self._tractogram)),
                                       np.array(indices_to_remove)).astype(int)

        tmp_streamlines = self.streamlines[indices_to_keep]
        tmp_data_per_point = self._tractogram.data_per_point[indices_to_keep]
        tmp_data_per_streamline =\
            self._tractogram.data_per_streamline[indices_to_keep]

        self._tractogram = Tractogram(tmp_streamlines.copy(),
                                      data_per_point=tmp_data_per_point,
                                      data_per_streamline=tmp_data_per_streamline,
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
                logger.debug('Moved streamlines from vox to voxmm')
        else:
            logger.warning('Wrong initial space for this function')
            return

    def _voxmm_to_vox(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.VOXMM:
            if self._tractogram.streamlines._data.size > 0:
                self._tractogram.streamlines._data /= np.asarray(
                    self._voxel_sizes)
                self._space = Space.VOX
                logger.debug('Moved streamlines from voxmm to vox')
        else:
            logger.warning('Wrong initial space for this function')
            return

    def _vox_to_rasmm(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.VOX:
            if self._tractogram.streamlines._data.size > 0:
                self._tractogram.apply_affine(self._affine)
                self._space = Space.RASMM
                logger.debug('Moved streamlines from vox to rasmm')
        else:
            logger.warning('Wrong initial space for this function')
            return

    def _rasmm_to_vox(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.RASMM:
            if self._tractogram.streamlines._data.size > 0:
                self._tractogram.apply_affine(self._inv_affine)
                self._space = Space.VOX
                logger.debug('Moved streamlines from rasmm to vox')
        else:
            logger.warning('Wrong initial space for this function')
            return

    def _voxmm_to_rasmm(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.VOXMM:
            if self._tractogram.streamlines._data.size > 0:
                self._tractogram.streamlines._data /= np.asarray(
                    self._voxel_sizes)
                self._tractogram.apply_affine(self._affine)
                self._space = Space.RASMM
                logger.debug('Moved streamlines from voxmm to rasmm')
        else:
            logger.warning('Wrong initial space for this function')
            return

    def _rasmm_to_voxmm(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.RASMM:
            if self._tractogram.streamlines._data.size > 0:
                self._tractogram.apply_affine(self._inv_affine)
                self._tractogram.streamlines._data *= np.asarray(
                    self._voxel_sizes)
                self._space = Space.VOXMM
                logger.debug('Moved streamlines from rasmm to voxmm')
        else:
            logger.warning('Wrong initial space for this function')
            return

    def _shift_voxel_origin(self):
        """ Unsafe function to switch the origin from center to corner
        and vice versa """
        if not self.streamlines:
            return

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
            logger.debug('Origin moved to the corner of voxel')
            self._origin = Origin.TRACKVIS
        else:
            logger.debug('Origin moved to the center of voxel')
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
        logger.error('data_per_point MUST be a dictionary')
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
                         'incoherent number of streamlines.')
            return False

        for values in data[key]:
            total_point_entries += len(values)

        if total_point_entries != total_point:
            logger.error('Missing entry for streamlines points data, '
                         'incoherent number of points per streamlines.')
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
        logger.error('data_per_point MUST be a dictionary')
        return False
    elif data == {}:
        return True

    total_streamline = 0
    for _ in streamlines:
        total_streamline += 1

    for key in data.keys():
        if not len(data[key]) == total_streamline:
            logger.error('Missing entry for streamlines points data, '
                         'incoherent number of streamlines.')
            return False

    return True
