import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, sin, cos

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float direct_flip_dist(float *a,float *b,
                            cnp.npy_intp rows) nogil:
    r''' Direct and flip average distance between two streamlines

    Parameters
    ----------
    a : float pointer
        first streamline
    b : float pointer
        second streamline
    rows : number of points of the streamline
        both tracks need to have the same number of points

    Returns
    -------
    out : float
        mininum of direct and flipped average distance added

    '''
    cdef:
        cnp.npy_intp i=0, j=0
        float sub=0, subf=0, distf=0, dist=0, tmprow=0, tmprowf=0


    for i in range(rows):
        tmprow = 0
        tmprowf = 0
        for j in range(3):
            sub = a[i * 3 + j] - b[i * 3 + j]
            subf = a[i * 3 + j] - b[(rows - 1 - i) * 3 + j]
            tmprow += sub * sub
            tmprowf += subf * subf
        dist += sqrt(tmprow)
        distf += sqrt(tmprowf)

    dist = dist / <float>rows 
    distf = distf / <float>rows

    if dist <= distf:
        return dist
    return distf


@cython.boundscheck(False)
@cython.wraparound(False)
def bundle_minimum_distance_rigid(float [:, ::1] static, 
                                  float [:, ::1] moving, 
                                  cnp.npy_intp static_size,
                                  cnp.npy_intp moving_size,
                                  cnp.npy_intp rows,
                                  float [:, ::1] D):
    """ MDF-based pairwise distance optimization function

    We minimize the distance between moving streamlines of the same number of
    points as they align with the static streamlines.

    Parameters
    -----------
    static: array
        Static streamlines

    moving: array
        Moving streamlines. These will be transform to align with
        the static streamlines

    Returns
    -------
    cost : double
    """

    cdef: 
        cnp.npy_intp i, j, mov_i=0, mov_j=0#, I, J, rows

    with nogil:

        for i in range(static_size):
            mov_j = 0
            for j in range(moving_size):

                D[i, j] = direct_flip_dist(&static[mov_i, 0],
                                           &moving[mov_j, 0],
                                           rows)
                
                mov_j += rows
            mov_i += rows
    
    return np.asarray(D)


