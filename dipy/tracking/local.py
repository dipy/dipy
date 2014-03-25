import numpy as np

from .localtrack import local_tracker
import utils

class LocalTracking(object):
    """A streamline generator for local tracking methods"""

    def __init__(self, direction_getter, tissue_classifier, seeds, affine,
                 step_size, max_cross=None, maxlen=500):
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
            indices of input data. An identity matrix can be used to generate
            streamlines in "voxel coordinates" as long as isotropic voxels were
            used to acquire the data.
        step_size : float
            Step size used for tracking.
        max_cross : int or None
            The maximum number of direction to track from each seed in crossing
            voxels. By default all initial directions are tracked.
        maxlen : int
            Maximum number of steps to track from seed. Used to prevent
            infinite loops.

        """
        self.direction_getter = direction_getter
        self.tissue_classifier = tissue_classifier
        self.seeds = seeds
        if affine.shape != (4, 4):
            raise ValueError("affine should be a (4, 4) array.")
        self.affine = affine
        self.step_size = step_size
        self.max_cross = max_cross
        self.maxlen = maxlen

    def __iter__(self):
        # Make tracks, move them to point space and return
        track = self._generate_streamlines()
        return utils.move_streamlines(track, self.affine)

    def _generate_streamlines(self):
        """A streamline generator"""
        N = self.maxlen
        dg = self.direction_getter
        tc = self.tissue_classifier
        ss = self.step_size
        max_cross = self.max_cross

        inv_A = np.linalg.inv(self.affine)
        lin = inv_A[:3, :3]
        offset = inv_A[:3, 3]

        # TODO: compute better voxel size
        vs = self.affine.diagonal()[:3]
        F = np.empty((N + 1, 3), dtype=float)
        B = F.copy()
        for s in self.seeds:
            s = np.dot(lin, s) + offset
            directions = dg.initial_direction(s)
            directions = directions[:max_cross]
            for first_step in directions:
                stepsF = local_tracker(dg, tc, s, first_step, vs, F, ss, 1)
                if stepsF < 0:
                    continue
                first_step = -first_step
                stepsB = local_tracker(dg, tc, s, first_step, vs, B, ss, 1)
                if stepsB < 0:
                    continue
                yield np.concatenate((B[stepsB-1:0:-1], F[:stepsF]), axis=0)

