cimport cython
cimport numpy as np
from .direction_getter cimport DirectionGetter
from .tissue_classifier cimport (TissueClassifier, TissueClass, TRACKPOINT,
                                 ENDPOINT, OUTSIDEIMAGE, INVALIDPOINT)


cdef void fixed_step(double *point, double *direction, double stepsize) nogil:
    for i in range(3):
        point[i] += direction[i] * stepsize


cdef inline void copypoint(double *a, double *b) nogil:
    for i in range(3):
        b[i] = a[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def local_tracker(DirectionGetter dg, TissueClassifier tc,
                  np.ndarray[np.float_t, ndim=1] seed,
                  np.ndarray[np.float_t, ndim=1] first_step,
                  np.ndarray[np.float_t, ndim=1] voxel_size,
                  np.ndarray[np.float_t, ndim=2, mode='c'] streamline,
                  double stepsize,
                  int fixedstep):

    if (seed.shape[0] != 3 or first_step.shape[0] != 3 or
        voxel_size.shape[0] != 3 or streamline.shape[1] != 3):
        raise ValueError()

    cdef:
        int i
        TissueClass tssuclass
        double point[3], dir[3], vs[3], voxdir[3]
        double[::1] pview = point, dview = dir

    for i in range(3):
        point[i] = seed[i]
        dir[i] = first_step[i]
        vs[i] = voxel_size[i]

    for i in range(streamline.shape[0]):
        copypoint(point, &streamline[i, 0])
        if dg.get_direction(pview, dview):
            i += 1
            break
        for j in range(3):
            voxdir[j] = dir[j] / vs[j]
        fixed_step(point, voxdir, stepsize)
        tssuclass = tc.check_point(pview)
        if tssuclass == TRACKPOINT:
            continue
        elif tssuclass == ENDPOINT:
            i += 1
            break
        elif tssuclass == OUTSIDEIMAGE:
            break
        elif tssuclass == INVALIDPOINT:
            i = - (i + 1)
            break
    return i

