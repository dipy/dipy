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
  Matrix multiplication between the matrix X and a vector beta.

  Parameters
  ----------
  row : int array (1-dimensional)
      Row indices.
  col : int array (1-dimensional)
      Column indices.
  X : float array (1-dimensional)
      The values of the matrix in these coordinates
  beta : float array (1-dimensional)
      The vector to be multiplied
  in_shape : int
      The shape of the 'input' (X).
  out_shape : int
      The shape of the 'output' (X@beta).

  Returns
  -------
  1-dimensional array : X@beta
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
  Matrix multiplication between the matrix transpose Xt and a vector beta.

  Parameters
  ----------
  row : int array (1-dimensional)
      Row indices.
  col : int array (1-dimensional)
      Column indices.
  X : float array (1-dimensional)
      The values of the matrix in these coordinates
  beta : float array (1-dimensional)
      The vector to be multiplied
  in_shape : int
      The shape of the 'input' (X).
  out_shape : int
      The shape of the 'output' (X.T@beta).

  Returns
  -------
  1-dimensional array : X.T@beta
  """
  ans = np.zeros(out_shape)

  cdef int i

  for i in range(in_shape):
    ans[col[i]] = ans[col[i]] + X[i] * beta[row[i]]

  return ans


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
cpdef gradient_change(row, col, X, beta, to_fit, shape1, shape2):
  """
  Calculate the change in gradient for optimization with a sparse array.

  Parameters
  ----------
  row : int array (1-dimensional)
      Row indices.
  col : int array (1-dimensional)
      Column indices.
  X : float array (1-dimensional)
      The values of the matrix in these coordinates
  beta : float array (1-dimensional)
      The current parameter setting

  Returns
  -------
  XtXby : float array

  Note
  ----
  For the gradient descent problem, in each step of the optimization:

     beta = beta - gradient

  where:

     gradient =  X.T @ (X@beta - y)

  It is this latter that is calculated here.
  """
  cdef Xb = spdot(row, col, X, beta, row.shape[0], shape1)
  cdef Xby = Xb - to_fit
  cdef XtXby = spdot_t(row, col, X, Xby, col.shape[0], shape2)
  return XtXby
