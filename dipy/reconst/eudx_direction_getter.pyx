# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False

cimport cython
cimport numpy as cnp
import numpy as np

from dipy.tracking.propspeed cimport _propagation_direction
from dipy.tracking.direction_getter cimport DirectionGetter

cdef extern from "dpy_math.h" nogil:
    double dpy_rint(double x)


cdef class EuDXDirectionGetter(DirectionGetter):
    """Deterministic Direction Getter based on peak directions.

    This class contains the cython portion of the code for PeaksAndMetrics and
    is not meant to be used on its own.
    """
    cdef:
        public double qa_thr, ang_thr, total_weight
        public double[:, :, :, ::1] _qa, _ind
        public double[:, ::1] _odf_vertices
        int initialized

    def __cinit__(self):
        initialized = False
        self.qa_thr = 0.0239
        self.ang_thr = 60
        self.total_weight = .5
        self.sphere = None
        self.peak_indices = None
        self.peak_values = None
        self.peak_dirs = None
        self.gfa = None
        self.qa = None
        self.shm_coeff = None
        self.B = None
        self.odf = None

    def _initialize(self):
        """First time that a PAM instance is used as a direction getter,
        initialize all the memoryviews.
        """
        if self.peak_values.shape != self.peak_indices.shape:
            msg = "shapes of peak_values and peak_indices do not match"
            raise ValueError(msg)
        self._qa = np.ascontiguousarray(self.peak_values, dtype=np.double)
        self._ind = np.ascontiguousarray(self.peak_indices, dtype=np.double)
        self._odf_vertices = np.asarray(
            self.sphere.vertices, dtype=np.double, order='C'
        )

        self.initialized = True

    cpdef cnp.ndarray[cnp.float_t, ndim=2] initial_direction(self,
                                                           double[::1] point):
        """The best starting directions for fiber tracking from point

        All the valid peaks in the voxel closest to point are returned as
        initial directions.

        """
        if not self.initialized:
            self._initialize()

        cdef:
            cnp.npy_intp numpeaks, i
            cnp.npy_intp ijk[3]

        # ijk is the closest voxel to point
        for i in range(3):
            ijk[i] = <cnp.npy_intp> dpy_rint(point[i])
            if ijk[i] < 0 or ijk[i] >= self._ind.shape[i]:
                raise IndexError("point outside data")

        # Check to see how many peaks were found in the voxel
        numpeaks=0
        for i in range(self.peak_values.shape[3]):
            # Test if the value associated to the peak is > 0
            if self._qa[ijk[0], ijk[1], ijk[2], i] > 0:
                numpeaks = numpeaks + 1
            else:
                break

        # Create directions array and copy peak directions from vertices
        res = np.empty((numpeaks, 3))
        for i in range(numpeaks):
            peak_index = self._ind[ijk[0], ijk[1], ijk[2], i]
            res[i, :] = self._odf_vertices[<cnp.npy_intp> peak_index, :]

        return res


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int get_direction_c(self, double[::1] point, double[::1] direction):
        """Interpolate closest peaks to direction from voxels neighboring point

        Update direction and return 0 if successful. If no tracking direction
        could be found, return 1.

        """
        if not self.initialized:
            self._initialize()

        cdef:
            cnp.npy_intp s
            double newdirection[3]
            cnp.npy_intp qa_shape[4]
            cnp.npy_intp qa_strides[4]

        for i in range(4):
            qa_shape[i] = self._qa.shape[i]
            qa_strides[i] = self._qa.strides[i]

        s = _propagation_direction(&point[0], &direction[0],
                                   &self._qa[0, 0, 0, 0],
                                   &self._ind[0, 0, 0, 0],
                                   &self._odf_vertices[0, 0], self.qa_thr,
                                   self.ang_thr, qa_shape, qa_strides,
                                   newdirection, self.total_weight)
        if s:
            for i in range(3):
                direction[i] = newdirection[i]
            return 0
        else:
            return 1
