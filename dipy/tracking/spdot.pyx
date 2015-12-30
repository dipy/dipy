import cython

import numpy as np
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
def spdot(cnp.ndarray[cnp.npy_intp, ndim=1] row,
          cnp.ndarray[cnp.npy_intp, ndim=1] col,
          cnp.ndarray[double, ndim=1] X,
          cnp.ndarray[double, ndim=1] beta,
          out_shape):
  """
  Matrix multiplication between the matrix X and a vector beta
  """
  ans = np.zeros(out_shape)

  cdef:
      cnp.npy_intp i

  for i in range(len(row)):
    ans[row[i]] = ans[row[i]] + X[i] * beta[col[i]]

  return ans

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
def spdot_t(cnp.ndarray[cnp.npy_intp, ndim=1] row,
            cnp.ndarray[cnp.npy_intp, ndim=1] col,
            cnp.ndarray[double, ndim=1] X,
            cnp.ndarray[double, ndim=1] beta,
            out_shape):
  """
  Matrix multiplication between the matrix Xt and a vector beta
  """
  ans = np.zeros(out_shape)

  cdef:
      cnp.npy_intp i

  for i in range(len(col)):
    ans[col[i]] = ans[col[i]] + X[i] * beta[row[i]]

  return ans
