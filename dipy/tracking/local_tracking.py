import random
from collections.abc import Iterable

import numpy as np

from dipy.tracking.localtrack import local_tracker, pft_tracker
from dipy.tracking.stopping_criterion import (AnatomicalStoppingCriterion,
                                              StreamlineStatus)
from dipy.tracking import utils


class LocalTracking(object):

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

    def __init__(self, direction_getter, stopping_criterion, seeds, affine,
                 step_size, max_cross=None, maxlen=500, fixedstep=True,
                 return_all=True, random_seed=None, save_seeds=False):
        """Creates streamlines by using local fiber-tracking.

        Parameters
        ----------
        direction_getter : instance of DirectionGetter
            Used to get directions for fiber tracking.
        stopping_criterion : instance of StoppingCriterion
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
        random_seed : int
            The seed for the random seed generator (numpy.random.seed and
            random.seed).
        save_seeds : bool
            If True, return seeds alongside streamlines
        """

        self.direction_getter = direction_getter
        self.stopping_criterion = stopping_criterion
        self.seeds = seeds
        if affine.shape != (4, 4):
            raise ValueError("affine should be a (4, 4) array.")
        if step_size <= 0:
            raise ValueError("step_size must be greater than 0.")
        if maxlen < 1:
            raise ValueError("maxlen must be greater than 0.")
        if not isinstance(seeds, Iterable):
            raise ValueError("seeds should be (N,3) array.")

        self.affine = affine
        self._voxel_size = np.ascontiguousarray(self._get_voxel_size(affine),
                                                dtype=float)
        self.step_size = step_size
        self.fixed_stepsize = fixedstep
        self.max_cross = max_cross
        self.max_length = maxlen
        self.return_all = return_all
        self.random_seed = random_seed
        self.save_seeds = save_seeds

    def _tracker(self, seed, first_step, streamline):
        return local_tracker(self.direction_getter,
                             self.stopping_criterion,
                             seed,
                             first_step,
                             self._voxel_size,
                             streamline,
                             self.step_size,
                             self.fixed_stepsize)

    def __iter__(self):
        # Make tracks, move them to point space and return
        track = self._generate_tractogram()

        return utils.transform_tracking_output(track, self.affine,
                                               save_seeds=self.save_seeds)

    def _generate_tractogram(self):
        """A streamline generator"""

        # Get inverse transform (lin/offset) for seeds
        inv_A = np.linalg.inv(self.affine)
        lin = inv_A[:3, :3]
        offset = inv_A[:3, 3]

        F = np.empty((self.max_length + 1, 3), dtype=float)
        B = F.copy()
        for s in self.seeds:
            s = np.dot(lin, s) + offset
            # Set the random seed in numpy and random
            if self.random_seed is not None:
                s_random_seed = hash(np.abs((np.sum(s)) + self.random_seed)) \
                    % (2**32 - 1)
                random.seed(s_random_seed)
                np.random.seed(s_random_seed)
            directions = self.direction_getter.initial_direction(s)
            if directions.size == 0 and self.return_all:
                # only the seed position
                if self.save_seeds:
                    yield [s], s
                else:
                    yield [s]
            directions = directions[:self.max_cross]
            for first_step in directions:
                stepsF, stream_status = self._tracker(s, first_step, F)
                if not (self.return_all or
                        stream_status == StreamlineStatus.ENDPOINT or
                        stream_status == StreamlineStatus.OUTSIDEIMAGE):
                    continue
                first_step = -first_step
                stepsB, stream_status = self._tracker(s, first_step, B)
                if not (self.return_all or
                        stream_status == StreamlineStatus.ENDPOINT or
                        stream_status == StreamlineStatus.OUTSIDEIMAGE):
                    continue
                if stepsB == 1:
                    streamline = F[:stepsF].copy()
                else:
                    parts = (B[stepsB - 1:0:-1], F[:stepsF])
                    streamline = np.concatenate(parts, axis=0)

                # move to the next streamline if only the seed position
                # and not return all
                if len(streamline) > 1 or self.return_all:
                    if self.save_seeds:
                        yield streamline, s
                    else:
                        yield streamline


class ParticleFilteringTracking(LocalTracking):

    def __init__(self, direction_getter, stopping_criterion, seeds, affine,
                 step_size, max_cross=None, maxlen=500,
                 pft_back_tracking_dist=2, pft_front_tracking_dist=1,
                 pft_max_trial=20, particle_count=15, return_all=True,
                 random_seed=None, save_seeds=False):
        r"""A streamline generator using the particle filtering tractography
        method [1]_.

        Parameters
        ----------
        direction_getter : instance of ProbabilisticDirectionGetter
            Used to get directions for fiber tracking.
        stopping_criterion : instance of AnatomicalStoppingCriterion
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
        pft_back_tracking_dist : float
            Distance in mm to back track before starting the particle filtering
            tractography. The total particle filtering tractography distance is
            equal to back_tracking_dist + front_tracking_dist.
            By default this is set to 2 mm.
        pft_front_tracking_dist : float
            Distance in mm to run the particle filtering tractography after the
            the back track distance. The total particle filtering tractography
            distance is equal to back_tracking_dist + front_tracking_dist. By
            default this is set to 1 mm.
        pft_max_trial : int
            Maximum number of trial for the particle filtering tractography
            (Prevents infinite loops).
        particle_count : int
            Number of particles to use in the particle filter.
        return_all : bool
            If true, return all generated streamlines, otherwise only
            streamlines reaching end points or exiting the image.
        random_seed : int
            The seed for the random seed generator (numpy.random.seed and
            random.seed).
        save_seeds : bool
            If True, return seeds alongside streamlines


        References
        ----------
        .. [1] Girard, G., Whittingstall, K., Deriche, R., & Descoteaux, M.
               Towards quantitative connectivity analysis: reducing
               tractography biases. NeuroImage, 98, 266-278, 2014.
        """

        if not isinstance(stopping_criterion, AnatomicalStoppingCriterion):
            raise ValueError("expecting AnatomicalStoppingCriterion")

        self.pft_max_nbr_back_steps = int(np.ceil(pft_back_tracking_dist
                                                  / step_size))
        self.pft_max_nbr_front_steps = int(np.ceil(pft_front_tracking_dist
                                                   / step_size))
        pft_max_steps = (self.pft_max_nbr_back_steps +
                         self.pft_max_nbr_front_steps)

        if (self.pft_max_nbr_front_steps < 0
                or self.pft_max_nbr_back_steps < 0
                or pft_max_steps < 1):
            raise ValueError("The number of PFT steps must be greater than 0.")

        if particle_count <= 0:
            raise ValueError("The particle count must be greater than 0.")

        self.directions = np.empty((maxlen + 1, 3), dtype=float)

        self.pft_max_trial = pft_max_trial
        self.particle_count = particle_count
        self.particle_paths = np.empty((2, self.particle_count,
                                        pft_max_steps + 1, 3),
                                       dtype=float)
        self.particle_weights = np.empty(self.particle_count, dtype=float)
        self.particle_dirs = np.empty((2, self.particle_count,
                                       pft_max_steps + 1, 3), dtype=float)
        self.particle_steps = np.empty((2, self.particle_count), dtype=int)
        self.particle_stream_statuses = np.empty((2, self.particle_count),
                                                 dtype=int)
        super(ParticleFilteringTracking, self).__init__(direction_getter,
                                                        stopping_criterion,
                                                        seeds,
                                                        affine,
                                                        step_size,
                                                        max_cross,
                                                        maxlen,
                                                        True,
                                                        return_all,
                                                        random_seed,
                                                        save_seeds)

    def _tracker(self, seed, first_step, streamline):
        return pft_tracker(self.direction_getter,
                           self.stopping_criterion,
                           seed,
                           first_step,
                           self._voxel_size,
                           streamline,
                           self.directions,
                           self.step_size,
                           self.pft_max_nbr_back_steps,
                           self.pft_max_nbr_front_steps,
                           self.pft_max_trial,
                           self.particle_count,
                           self.particle_paths,
                           self.particle_dirs,
                           self.particle_weights,
                           self.particle_steps,
                           self.particle_stream_statuses)
