import cython

import numpy as np
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
cpdef cnp.ndarray[double, ndim=1] spdot(cnp.ndarray[cnp.npy_intp, ndim=1] row,
                                        cnp.ndarray[cnp.npy_intp, ndim=1] col,
                                        cnp.ndarray[double, ndim=1] X,
                                        cnp.ndarray[double, ndim=1] beta,
                                        int in_shape,
                                        int out_shape):
  """
  Matrix multiplication between the matrix X and a vector beta
  """
  ans = np.zeros(out_shape)

  cdef int i

  for i in range(in_shape):
    ans[row[i]] = ans[row[i]] + X[i] * beta[col[i]]

  return ans

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
cpdef cnp.ndarray[double, ndim=1] spdot_t(cnp.ndarray[cnp.npy_intp, ndim=1] row,
                                          cnp.ndarray[cnp.npy_intp, ndim=1] col,
                                          cnp.ndarray[double, ndim=1] X,
                                          cnp.ndarray[double, ndim=1] beta,
                                          int in_shape,
                                          int out_shape):
  """
  Matrix multiplication between the matrix Xt and a vector beta
  """
  ans = np.zeros(out_shape)

  cdef int i

  for i in range(in_shape):
    ans[col[i]] = ans[col[i]] + X[i] * beta[row[i]]

  return ans


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
cpdef gradient_change(row, col, sig, beta, to_fit, shape1, shape2):
  """
  Calculate the change in gradient for optimization with a sparse array.

  parameters
  ----------
  """
  cdef Xh = spdot(row, col, sig, beta, row.shape[0], shape1)
  cdef margin = Xh - to_fit
  cdef XtXh = spdot_t(row, col, sig, margin, col.shape[0], shape2)
  return XtXh
