# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: wraparound=False

cdef extern from "dpy_math.h" nogil:
    int dpy_rint(double)

from dipy.utils.fast_numpy cimport random, RNGState, random_float
from dipy.core.interpolation cimport trilinear_interpolate4d_c

import numpy as np

cdef class StoppingCriterion:
    cpdef StreamlineStatus check_point(self, double[::1] point):
        if point.shape[0] != 3:
            raise ValueError("Point has wrong shape")

        return self.check_point_c(&point[0])

    cdef StreamlineStatus check_point_c(self, double* point, RNGState* rng=NULL) noexcept nogil:
         pass


cdef class BinaryStoppingCriterion(StoppingCriterion):
    """
    cdef:
        unsigned char[:, :, :] mask
    """

    def __cinit__(self, mask):
        self.mask = (mask > 0).astype('uint8')

    cdef StreamlineStatus check_point_c(self, double* point, RNGState* rng=NULL) noexcept nogil:
        cdef:
            unsigned char result
            int err
            int voxel[3]

        voxel[0] = int(dpy_rint(point[0]))
        voxel[1] = int(dpy_rint(point[1]))
        voxel[2] = int(dpy_rint(point[2]))

        if (voxel[0] < 0 or voxel[0] >= self.mask.shape[0]
                or voxel[1] < 0 or voxel[1] >= self.mask.shape[1]
                or voxel[2] < 0 or voxel[2] >= self.mask.shape[2]):
            return OUTSIDEIMAGE

        result = self.mask[voxel[0], voxel[1], voxel[2]]

        if result > 0:
            return TRACKPOINT
        else:
            return ENDPOINT


cdef class ThresholdStoppingCriterion(StoppingCriterion):

    def __cinit__(self, metric_map, double threshold):
        self.metric_map = np.asarray(metric_map, 'float64')
        self.threshold = threshold

    cdef StreamlineStatus check_point_c(self, double* point, RNGState* rng=NULL) noexcept nogil:
        cdef:
            double result
            int err
            double interp_out_double[1]

        err = trilinear_interpolate4d_c(self.metric_map[..., None],
                                        point,
                                        &interp_out_double[0])
        if err == -1:
            return OUTSIDEIMAGE
        elif err != 0:
            # This should never happen
            raise RuntimeError(
                "Unexpected interpolation error (code:%i)" % err)

        result = interp_out_double[0]

        if result > self.threshold:
            return TRACKPOINT
        else:
            return ENDPOINT


cdef class AnatomicalStoppingCriterion(StoppingCriterion):
    r"""
    Abstract class that takes as input included and excluded tissue maps.
    The 'include_map' defines when the streamline reached a 'valid' stopping
    region (e.g. gray matter partial volume estimation (PVE) map) and the
    'exclude_map' defines when the streamline reached an 'invalid' stopping
    region (e.g. corticospinal fluid PVE map). The background of the anatomical
    image should be added to the 'include_map' to keep streamlines exiting the
    brain (e.g. through the brain stem).

    cdef:
        double[:, :, :] include_map, exclude_map

    """
    def __cinit__(self, include_map, exclude_map, *args, **kw):
        self.include_map = np.asarray(include_map, 'float64')
        self.exclude_map = np.asarray(exclude_map, 'float64')

    @classmethod
    def from_pve(cls, wm_map, gm_map, csf_map, **kw):
        """AnatomicalStoppingCriterion from partial volume fraction (PVE) maps.

        Parameters
        ----------
        wm_map : array
            The partial volume fraction of white matter at each voxel.
        gm_map : array
            The partial volume fraction of gray matter at each voxel.
        csf_map : array
            The partial volume fraction of corticospinal fluid at each
            voxel.

        """
        # include map = gray matter + image background
        include_map = np.copy(gm_map)
        include_map[(wm_map + gm_map + csf_map) == 0] = 1
        # exclude map = csf
        exclude_map = np.copy(csf_map)
        return cls(include_map, exclude_map, **kw)

    cpdef double get_exclude(self, double[::1] point):
        if point.shape[0] != 3:
            raise ValueError("Point has wrong shape")

        return self.get_exclude_c(&point[0])

    cdef get_exclude_c(self, double* point):
        cdef:
            double interp_out_double[1]

        exclude_err = trilinear_interpolate4d_c(self.exclude_map[..., None],
                                                point,
                                                &interp_out_double[0])
        if exclude_err != 0:
            return 0
        return interp_out_double[0]

    cpdef double get_include(self, double[::1] point):
        if point.shape[0] != 3:
            raise ValueError("Point has wrong shape")

        return self.get_include_c(&point[0])

    cdef get_include_c(self, double* point):
        cdef:
            double interp_out_double[1]

        include_err = trilinear_interpolate4d_c(self.include_map[..., None],
                                                point,
                                                &interp_out_double[0])
        if include_err != 0:
            return 0
        return interp_out_double[0]


cdef class ActStoppingCriterion(AnatomicalStoppingCriterion):
    """Anatomically-Constrained Tractography (ACT) stopping criterion.

    See :footcite:p:`Smith2012` for further details about the method.

    This implements the use of partial volume fraction (PVE) maps to
    determine when the tracking stops. The proposed method
    :footcite:p:`Smith2012` that cuts streamlines going through subcortical gray
    matter regions is not implemented here. The backtracking technique for
    streamlines reaching INVALIDPOINT is not implemented either.

    References
    ----------
    .. footbibliography::
    """

    def __cinit__(self, include_map, exclude_map):
        self.include_map = np.asarray(include_map, 'float64')
        self.exclude_map = np.asarray(exclude_map, 'float64')

    cdef StreamlineStatus check_point_c(self, double* point, RNGState* rng=NULL) noexcept nogil:
        cdef:
            double include_result, exclude_result
            int include_err, exclude_err
            double interp_out_double[1]

        include_err = trilinear_interpolate4d_c(self.include_map[..., None],
                                                point,
                                                &interp_out_double[0])
        include_result = interp_out_double[0]

        exclude_err = trilinear_interpolate4d_c(self.exclude_map[..., None],
                                                point,
                                                &interp_out_double[0])
        exclude_result = interp_out_double[0]

        if include_err == -1 or exclude_err == -1:
            return OUTSIDEIMAGE
        elif include_err != 0:
            # This should never happen
            raise RuntimeError("Unexpected interpolation error " +
                               "(include_map - code:%i)" % include_err)
        elif exclude_err != 0:
            # This should never happen
            raise RuntimeError("Unexpected interpolation error " +
                               "(exclude_map - code:%i)" % exclude_err)

        if include_result > 0.5:
            return ENDPOINT
        elif exclude_result > 0.5:
            return INVALIDPOINT
        else:
            return TRACKPOINT


cdef class CmcStoppingCriterion(AnatomicalStoppingCriterion):
    r"""
    Continuous map criterion (CMC) stopping criterion.

    This implements the use of partial volume fraction (PVE) maps to
    determine when the tracking stops :footcite:p:`Girard2014`.

    cdef:
        double[:, :, :] include_map, exclude_map
        double step_size
        double average_voxel_size
        double correction_factor

    References
    ----------
    .. footbibliography::
    """

    def __cinit__(self, include_map, exclude_map, step_size, average_voxel_size):
        self.step_size = step_size
        self.average_voxel_size = average_voxel_size
        self.correction_factor = step_size / average_voxel_size

    cdef StreamlineStatus check_point_c(self, double* point, RNGState* rng=NULL) noexcept nogil:
        cdef:
            double include_result, exclude_result
            int include_err, exclude_err
            double p
            double interp_out_double[1]

        include_err = trilinear_interpolate4d_c(self.include_map[..., None],
                                                point,
                                                &interp_out_double[0])
        include_result = interp_out_double[0]

        exclude_err = trilinear_interpolate4d_c(self.exclude_map[..., None],
                                                point,
                                                &interp_out_double[0])
        exclude_result = interp_out_double[0]

        if include_err == -1 or exclude_err == -1:
            return OUTSIDEIMAGE
        elif include_err == -2 or exclude_err == -2:
            raise ValueError("Point has wrong shape")
        elif include_err != 0:
            # This should never happen
            raise RuntimeError("Unexpected interpolation error " +
                               "(include_map - code:%i)" % include_err)
        elif exclude_err != 0:
            # This should never happen
            raise RuntimeError("Unexpected interpolation error " +
                               "(exclude_map - code:%i)" % exclude_err)

        # rng = np.random.default_rng()

        # test if the tracking continues
        if include_result + exclude_result <= 0:
            return TRACKPOINT
        num = max(0, (1 - include_result - exclude_result))
        den = num + include_result + exclude_result
        p = (num / den) ** self.correction_factor
        if rng != NULL:
            if random_float(rng) < p:
                return TRACKPOINT

            # test if the tracking stopped in the include tissue map
            p = (include_result / (include_result + exclude_result))
            if random_float(rng) < p:
                return ENDPOINT
        else:
            if random() < p:
                return TRACKPOINT

            # test if the tracking stopped in the include tissue map
            p = (include_result / (include_result + exclude_result))
            if random() < p:
                return ENDPOINT

        # the tracking stopped in the exclude tissue map
        return INVALIDPOINT
