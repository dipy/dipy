from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.linalg import svd


__all__ = ['polar']


def polar(a, side="right"):
    """
    Compute the polar decomposition.

    Returns the factors of the polar decomposition [1]_ `u` and `p` such
    that ``a = up`` (if `side` is "right") or ``a = pu`` (if `side` is
    "left"), where `p` is positive semidefinite.  Depending on the shape
    of `a`, either the rows or columns of `u` are orthonormal.  When `a`
    is a square array, `u` is a square unitary array.  When `a` is not
    square, the "canonical polar decomposition" [2]_ is computed.

    Parameters
    ----------
    a : array_like, shape (m, n).
        The array to be factored.
    side : {'left', 'right'}, optional
        Determines whether a right or left polar decomposition is computed.
        If `side` is "right", then ``a = up``.  If `side` is "left",  then
        ``a = pu``.  The default is "right".

    Returns
    -------
    u : ndarray, shape (m, n)
        If `a` is square, then `u` is unitary.  If m > n, then the columns
        of `a` are orthonormal, and if m < n, then the rows of `u` are
        orthonormal.
    p : ndarray
        `p` is Hermitian positive semidefinite.  If `a` is nonsingular, `p`
        is positive definite.  The shape of `p` is (n, n) or (m, m), depending
        on whether `side` is "right" or "left", respectively.

    References
    ----------
    .. [1] R. A. Horn and C. R. Johnson, "Matrix Analysis", Cambridge
           University Press, 1985.
    .. [2] N. J. Higham, "Functions of Matrices: Theory and Computation",
           SIAM, 2008.

    Notes
    -----
    Copyright (c) 2001, 2002 Enthought, Inc.
    All rights reserved.

    Copyright (c) 2003-2012 SciPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
    b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
    c. Neither the name of Enthought nor the names of the SciPy Developers
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
    OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGE.
    """
    if side not in ['right', 'left']:
        raise ValueError("`side` must be either 'right' or 'left'")
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("`a` must be a 2-D array.")

    w, s, vh = svd(a, full_matrices=False)
    u = w.dot(vh)
    if side == 'right':
        # a = up
        p = (vh.T.conj() * s).dot(vh)
    else:
        # a = pu
        p = (w * s).dot(w.T.conj())
    return u, p
