cimport cython
cimport numpy as np

cdef extern from "dpy_math.h" nogil:
    int dpy_rint(double)

from .interpolation cimport(trilinear_interpolate4d,
                            _trilinear_interpolate_c_4d)

import numpy as np

cdef class TissueClassifier:
    cpdef TissueClass check_point(self, double[::1] point) except PYERROR:
        pass


cdef class BinaryTissueClassifier(TissueClassifier):
    """
    cdef:
        unsigned char[:, :, :] mask
    """

    def __cinit__(self, mask):
        self.interp_out_view = self.interp_out_double
        self.mask = (mask > 0).astype('uint8')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef TissueClass check_point(self, double[::1] point) except PYERROR:
        cdef:
            unsigned char result
            int err
            int voxel[3]

        if point.shape[0] != 3:
            raise ValueError("Point has wrong shape")

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


cdef class ThresholdTissueClassifier(TissueClassifier):
    """
    # Declarations from tissue_classifier.pxd bellow
    cdef:
        double threshold, interp_out_double[1]
        double[:]  interp_out_view = interp_out_view
        double[:, :, :] metric_map
    """

    def __cinit__(self, metric_map, threshold):
        self.interp_out_view = self.interp_out_double
        self.metric_map = np.asarray(metric_map, 'float64')
        self.threshold = threshold

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef TissueClass check_point(self, double[::1] point) except PYERROR:
        cdef:
            double result
            int err

        err = _trilinear_interpolate_c_4d(self.metric_map[..., None], point,
                                          self.interp_out_view)
        if err == -1:
            return OUTSIDEIMAGE
        elif err == -2:
            raise ValueError("Point has wrong shape")
        elif err != 0:
            # This should never happen
            raise RuntimeError(
                "Unexpected interpolation error (code:%i)" % err)

        result = self.interp_out_view[0]

        if result > self.threshold:
            return TRACKPOINT
        else:
            return ENDPOINT


cdef class ActTissueClassifier(TissueClassifier):
    r"""
    Anatomically-Constrained Tractography (ACT) stopping criteria from [1]_.
    This implements the use of partial volume fraction (PVE) maps to
    determine when the tracking stops. The proposed ([1]_) method that
    cuts streamlines going through subcortical gray matter regions is
    not implemented here. The backtracking technique for
    streamlines reaching INVALIDPOINT is not implemented either.
    cdef:
        double interp_out_double[1]
        double[:]  interp_out_view = interp_out_view
        double[:, :, :] include_map, exclude_map

    References
    ----------
    .. [1] Smith, R. E., Tournier, J.-D., Calamante, F., & Connelly, A.
    "Anatomically-constrained tractography: Improved diffusion MRI
    streamlines tractography through effective use of anatomical
    information." NeuroImage, 63(3), 1924–1938, 2012.
    """

    def __cinit__(self, include_map, exclude_map):
        self.interp_out_view = self.interp_out_double
        self.include_map = np.asarray(include_map, 'float64')
        self.exclude_map = np.asarray(exclude_map, 'float64')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef TissueClass check_point(self, double[::1] point) except PYERROR:
        cdef:
            double include_result, exclude_result
            int include_err, exclude_err

        include_err = _trilinear_interpolate_c_4d(self.include_map[..., None],
                                                  point, self.interp_out_view)
        include_result = self.interp_out_view[0]

        exclude_err = _trilinear_interpolate_c_4d(self.exclude_map[..., None],
                                                  point, self.interp_out_view)
        exclude_result = self.interp_out_view[0]

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

        if include_result > 0.5:
            return ENDPOINT
        elif exclude_result > 0.5:
            return INVALIDPOINT
        else:
            return TRACKPOINT
