#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, sin, cos


cdef double direct_flip_dist(double *a,double *b,
                             cnp.npy_intp rows) nogil:
    r""" Direct and flip average distance between two streamlines

    Parameters
    ----------
    a : double pointer
        first streamline
    b : double pointer
        second streamline
    rows : number of points of the streamline
        both tracks need to have the same number of points

    Returns
    -------
    out : double
        mininum of direct and flipped average distances

    """
    cdef:
        cnp.npy_intp i=0, j=0
        double sub=0, subf=0, distf=0, dist=0, tmprow=0, tmprowf=0


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

    dist = dist / <double>rows
    distf = distf / <double>rows

    if dist <= distf:
        return dist
    return distf


def _bundle_minimum_distance_rigid(double [:, ::1] static,
                                   double [:, ::1] moving,
                                   cnp.npy_intp static_size,
                                   cnp.npy_intp moving_size,
                                   cnp.npy_intp rows,
                                   double [:, ::1] D):
    """ MDF-based pairwise distance optimization function

    We minimize the distance between moving streamlines of the same number of
    points as they align with the static streamlines.

    Parameters
    -----------
    static: array
        Static streamlines

    moving: array
        Moving streamlines 

    Returns
    -------
    cost : double
    """

    cdef:
        cnp.npy_intp i, j, mov_i=0, mov_j=0

    with nogil:

        for i in prange(static_size):
            
            for j in prange(moving_size):

                D[i, j] = direct_flip_dist(&static[i * rows, 0],
                                           &moving[j * rows, 0],
                                           rows)

    return np.asarray(D)


def _bundle_minimum_distance_rigid_nomat(double [:, ::1] stat,
                                         double [:, ::1] mov,
                                         cnp.npy_intp static_size,
                                         cnp.npy_intp moving_size,
                                         cnp.npy_intp rows):
    """ MDF-based pairwise distance optimization function

    We minimize the distance between moving streamlines of the same number of
    points as they align with the static streamlines.

    Parameters
    -----------
    static: array
        Static streamlines

    moving: array
        Moving streamlines 

    Returns
    -------
    cost : double

    Notes
    -----
    The difference withgit asd this function is that it is not saving the 
    """

    cdef:
        cnp.npy_intp i, j
        double sum_i, sum_j, min_j, min_i, tmp
        double inf = np.finfo('f4').max
        double dist

    with nogil:

        sum_i = 0

        for i in range(static_size):
            
            min_j = inf

            for j in range(moving_size):

                tmp = <double>direct_flip_dist(&stat[i * rows, 0], 
                                               &mov[j * rows, 0], rows)
                if tmp < min_j:
                    min_j = tmp

            sum_i += min_j

        sum_j = 0

        for j in range(moving_size):
            
            min_i = inf

            for i in range(static_size):

                tmp = direct_flip_dist(&stat[i * rows, 0], 
                                       &mov[j * rows, 0], rows)

                if tmp < min_i:
                    min_i = tmp

            sum_j += min_i

    dist = (sum_i/<double>static_size  + sum_j/<double>moving_size)
    
    return 0.25 * dist * dist


def _bundle_minimum_distance_rigid_nomat_parallel(double [:, ::1] stat,
                                                  double [:, ::1] mov,
                                                  cnp.npy_intp static_size,
                                                  cnp.npy_intp moving_size,
                                                  cnp.npy_intp rows):
    """ MDF-based pairwise distance optimization function

    We minimize the distance between moving streamlines of the same number of
    points as they align with the static streamlines.

    Parameters
    -----------
    static: array
        Static streamlines

    moving: array
        Moving streamlines 

    Returns
    -------
    cost : double

    Notes
    -----
    The difference withgit asd this function is that it is not saving the 
    """

    cdef:
        cnp.npy_intp i, j
        double sum_i, sum_j, min_j, min_i, tmp
        double inf = np.finfo('f4').max
        double dist


    with nogil:

        sum_i = 0

        for i in prange(static_size):
            
            min_j = inf

            for j in range(moving_size):

                tmp = <double>direct_flip_dist(&stat[i * rows, 0], 
                                               &mov[j * rows, 0], rows)
                if tmp < min_j:
                    min_j = tmp

            sum_i += min_j

        sum_j = 0

        for j in prange(moving_size):
            
            min_i = inf

            for i in range(static_size):

                tmp = <double> direct_flip_dist(&stat[i * rows, 0], 
                                                &mov[j * rows, 0], rows)

                if tmp < min_i:
                    min_i = tmp

            sum_j += min_i

        dist = (sum_i/<double>static_size  + sum_j/<double>moving_size)

        dist = 0.25 * dist * dist
    
    return dist


