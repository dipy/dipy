import numpy as np

#from dipy.tracking.local.tissue_classifier import ConstrainedTissueClassifier
from dipy.tracking.local.localtrack import local_tracker, pft_tracker

from dipy.align import Bunch
#from dipy.direction import ProbabilisticDirectionGetter
from dipy.tracking import utils


# enum TissueClass (tissue_classifier.pxd) is not accessible
# from here. To be changed when minimal cython version > 0.21.
# cython 0.21 - cpdef enum to export values into Python-level namespace
# https://github.com/cython/cython/commit/50133b5a91eea348eddaaad22a606a7fa1c7c457
TissueTypes = Bunch(OUTSIDEIMAGE=-1, INVALIDPOINT=0, TRACKPOINT=1, ENDPOINT=2)


class LocalTracking(object):
    """A streamline generator for local tracking methods"""

    @staticmethod
    def _get_voxel_size(affine):
        """Computes the voxel sizes of an image from the affine.

        Checks that the affine does not have any shear because local_tracker
        assumes that the data is sampled on a regular grid.

        """
        lin = affine[:3, :3]
        dotlin = np.dot(lin.T, lin)
        # Check that the affine is well behaved
        if not np.allclose(np.triu(dotlin, 1), 0., atol=1e-5):
            msg = ("The affine provided seems to contain shearing, data must "
                   "be acquired or interpolated on a regular grid to be used "
                   "with `LocalTracking`.")
            raise ValueError(msg)
        return np.sqrt(dotlin.diagonal())

    def __init__(self, direction_getter, tissue_classifier, seeds, affine,
                 step_size, max_cross=None, maxlen=500, fixedstep=True,
                 return_all=True):
        """Creates streamlines by using local fiber-tracking.

        Parameters
        ----------
        direction_getter : instance of DirectionGetter
            Used to get directions for fiber tracking.
        tissue_classifier : instance of TissueClassifier
            Identifies endpoints and invalid points to inform tracking.
        seeds : array (N, 3)
            Points to seed the tracking. Seed points should be given in point
            space of the track (see ``affine``).
        affine : array (4, 4)
            Coordinate space for the streamline point with respect to voxel
            indices of input data. This affine can contain scaling, rotational,
            and translational components but should not contain any shearing.
            An identity matrix can be used to generate streamlines in "voxel
            coordinates" as long as isotropic voxels were used to acquire the
            data.
        step_size : float
            Step size used for tracking.
        max_cross : int or None
            The maximum number of direction to track from each seed in crossing
            voxels. By default all initial directions are tracked.
        maxlen : int
            Maximum number of steps to track from seed. Used to prevent
            infinite loops.
        fixedstep : bool
            If true, a fixed stepsize is used, otherwise a variable step size
            is used.
        return_all : bool
            If true, return all generated streamlines, otherwise only
            streamlines reaching end points or exiting the image.
        """

        self.direction_getter = direction_getter
        self.tissue_classifier = tissue_classifier
        self.seeds = seeds
        if affine.shape != (4, 4):
            raise ValueError("affine should be a (4, 4) array.")
        self.affine = affine
        self._voxel_size = self._get_voxel_size(affine)
        self.step_size = step_size
        self.fixed_stepsize = fixedstep
        self.max_cross = max_cross
        self.max_length = maxlen
        self.return_all = return_all

    def _tracker(self, seed, first_step, streamline):
        return local_tracker(self.direction_getter,
                             self.tissue_classifier,
                             seed,
                             first_step,
                             self._voxel_size,
                             streamline,
                             self.step_size,
                             self.fixed_stepsize)

    def __iter__(self):
        # Make tracks, move them to point space and return
        track = self._generate_streamlines()
        return utils.move_streamlines(track, self.affine)

    def _generate_streamlines(self):
        """A streamline generator"""

        # Get inverse transform (lin/offset) for seeds
        inv_A = np.linalg.inv(self.affine)
        lin = inv_A[:3, :3]
        offset = inv_A[:3, 3]

        F = np.empty((self.max_length + 1, 3), dtype=float)
        B = F.copy()
        for s in self.seeds:
            s = np.dot(lin, s) + offset
            directions = self.direction_getter.initial_direction(s)
            if directions.size == 0 and self.return_all:
                # only the seed position
                yield [s]
            directions = directions[:self.max_cross]
            for first_step in directions:
                stepsF, tissue_class = self._tracker(s, first_step, F)
                if not (self.return_all or
                        tissue_class == TissueTypes.ENDPOINT or
                        tissue_class == TissueTypes.OUTSIDEIMAGE):
                    continue
                first_step = -first_step
                stepsB, tissue_class = self._tracker(s, first_step, B)
                if not (self.return_all or
                        tissue_class == TissueTypes.ENDPOINT or
                        tissue_class == TissueTypes.OUTSIDEIMAGE):
                    continue

                if stepsB == 1:
                    streamline = F[:stepsF].copy()
                else:
                    parts = (B[stepsB-1:0:-1], F[:stepsF])
                    streamline = np.concatenate(parts, axis=0)
                yield streamline


class ParticleFilteringTracking(LocalTracking):
    """A streamline generator using the particle filtering tractography method
    """

    def __init__(self, direction_getter, tissue_classifier, seeds, affine,
                 step_size, max_cross=None, maxlen=500, return_all=True,
                 max_pft_trial=20, back_tracking_dist=2, pft_tracking_dist=3):
        """Creates streamlines by using local fiber-tracking.

        Parameters
        ----------
        direction_getter : instance of ProbabilisticDirectionGetter
            Used to get directions for fiber tracking.
        tissue_classifier : instance of ConstrainedTissueClassifier
            Identifies endpoints and invalid points to inform tracking.
        seeds : array (N, 3)
            Points to seed the tracking. Seed points should be given in point
            space of the track (see ``affine``).
        affine : array (4, 4)
            Coordinate space for the streamline point with respect to voxel
            indices of input data. This affine can contain scaling, rotational,
            and translational components but should not contain any shearing.
            An identity matrix can be used to generate streamlines in "voxel
            coordinates" as long as isotropic voxels were used to acquire the
            data.
        step_size : float
            Step size used for tracking.
        back_tracking_dist : float
            Distance in mm to back track before starting the particle filtering
            tractography.
        pft_tracking_dist : float
            Distance in mm to run the particle filtering tractography.
        max_pft_trial : int
            Maximum number of trial for the particle filtering tractography
            (Prevents infinit loops).
        max_cross : int or None
            The maximum number of direction to track from each seed in crossing
            voxels. By default all initial directions are tracked.
        maxlen : int
            Maximum number of steps to track from seed. Used to prevent
            infinite loops.
        return_all : bool
            If true, return all generated streamlines, otherwise only
            streamlines reaching end points or exiting the image.
        """

        #if not isinstance(direction_getter, ProbabilisticDirectionGetter):
        #    raise ValueError("expecting ProbabilisticDirectionGetter")

        #if not isinstance(tissue_classifier, ConstrainedTissueClassifier):
        #    raise ValueError("expecting ConstrainedTissueClassifier")

        self.back_tracking_steps = int(np.ceil(back_tracking_dist / step_size))
        self.pft_tracking_steps = int(np.ceil(pft_tracking_dist / step_size))
        self.max_pft_trial = max_pft_trial
        self.nbr_particles = 10
        self.directions = np.empty((maxlen + 1, 3), dtype=float)
        super(ParticleFilteringTracking, self).__init__(direction_getter,
                                                        tissue_classifier,
                                                        seeds,
                                                        affine,
                                                        step_size,
                                                        max_cross,
                                                        maxlen,
                                                        True,
                                                        return_all)

    def _tracker(self, seed, first_step, streamline):
        return pft_tracker(seed,
                           first_step,
                           streamline,
                           self.direction_getter,
                           self.tissue_classifier,
                           self._voxel_size,
                           self.step_size,
                           self.directions,
                           self.back_tracking_steps,
                           self.pft_tracking_steps,
                           self.max_pft_trial,
                           self.nbr_particles)
