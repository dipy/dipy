# cython: boundcheck=False, wraparound=False, cdivision=True

# from __future__ import division

import cmath
import operator
from libc.math cimport fabs, tgamma, exp, floor
import numpy as np
cimport numpy as np
cimport cython

cdef double eps = np.finfo(np.float64).eps

cdef double hypsum1f1(double a, int b, double z, int maxterms=6000):

    cdef:
        int k = 0
        double s = 1., t = 1.

    while True:

        t *= a + k
        t /= b + k
        k += 1
        t /= k
        t *= z
        s += t

        if fabs(t) < eps:
            return s

        # Did not converge
        if k > maxterms:
            return -np.inf


cdef double hypsum2f0(double a, double b, double z, int maxterms=6000):

    cdef:
        int k = 0
        double s = 1., t = 1.,

    while True:

        t *= a + k
        t *= b + k
        k += 1
        t /= k
        t *= z
        s += t

        if fabs(t) < eps:
            return s

        # Did not converge
        if k > maxterms:
            return -np.inf


cdef double mag(z):
        return np.frexp(fabs(z))[1]


cdef power(a, b):
    try:
        return operator.pow(float(a), float(b))
    except (TypeError, ValueError):
        return operator.pow(complex(a), complex(b))


cdef bint isnpint(x):
    if type(x) is complex:
        if x.imag:
            return False
        x = x.real
    return x <= 0.0 and floor(x) == x


cdef nint_distance(z):
    cdef int n = int(z.real)

    if n == z:
        return n, -np.inf
    return n, mag(fabs(z-n))


cdef _check_need_perturb(terms, bint discard_known_zeros):

    cdef:
        bint perturb = False
        bint have_singular_nongamma_weight = False
        int n, i, k, term_index, data_index
        double z, d, x
        # double w_s, c_s, alpha_s, beta_s, a_s, b_s, z, d, x
        # double [::1] discard = np.array([], dtype=np.float64)
        # double [::1] term = np.array([], dtype=np.float64)
        # double [::1] data = np.array([], dtype=np.float64)
        int [::1] pole_count = np.zeros(3, dtype=np.int32)

    discard = []
    for term_index, term in enumerate(terms):
        w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term
        # Avoid division by zero in leading factors (TODO:
        # also check for near division by zero?)
        for k, w in enumerate(w_s):
            if not w:
                if np.real(c_s[k]) <= 0 and c_s[k]:
                    perturb = True
                    have_singular_nongamma_weight = True
        pole_count = np.zeros(3, dtype=np.int32)
        # Check for gamma and series poles and near-poles
        for data_index, data in enumerate([alpha_s, beta_s, b_s]):
            for i, x in enumerate(data):
                n, d = nint_distance(x)
                # Poles
                if n > 0:
                    continue
                if d == -np.inf:
                    # OK if we have a polynomial
                    # ------------------------------
                    if data_index == 2:
                        for u in a_s:
                            if isnpint(u) and u >= int(n):
                                break
                    else:
                        pole_count[data_index] += 1
        if (discard_known_zeros and
            pole_count[1] > pole_count[0] + pole_count[2] and
            not have_singular_nongamma_weight):
            discard.append(term_index)
        elif np.sum(pole_count):
            perturb = True
    return perturb, discard


cdef double hypercomb(function, params, bint discard_known_zeros=True):
    cdef:
        bint perturb
        int k
        double h = 3.0517578125e-05
        double v, z
        # list evaluated_terms
        # double w_s, c_s, alpha_s, beta_s, a_s, b_s, z
        # double evaluated_terms = np.array([], dtype=np.float64)

    terms = function(*params)
    perturb, discard =  _check_need_perturb(terms, discard_known_zeros)
    if perturb:
        for k in range(len(params)):
            params[k] += h
            # Heuristically ensure that the perturbations
            # are "independent" so that two perturbations
            # don't accidentally cancel each other out
            # in a subtraction.
            h += h / (k+1)
        terms = function(*params)
    if discard_known_zeros:
        terms = [term for (i, term) in enumerate(terms) if i not in discard]
    if not terms:
        return 0.
    evaluated_terms = []
    for term_index, term_data in enumerate(terms):
        w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term_data
        # Always hyp2f0
        assert len(a_s) == 2
        assert len(b_s) == 0
        v = np.prod([hypsum2f0(a_s[0], a_s[1], z)] + \
            [tgamma(a) for a in alpha_s] + \
            [1/tgamma(b) for b in beta_s] + \
            [power(w, c) for (w,c) in zip(w_s,c_s)])

        # hypsum2f0 did not converge
        if np.isinf(v):
            return v

        evaluated_terms.append(v)

    if len(terms) == 1 and (not perturb):
        return evaluated_terms[0]

    return np.sum(evaluated_terms)


def hyp1f1(a, b, z):

    magz = mag(z)

    if magz >= 7:
        # try:

            def h(a, b):
                E = cmath.exp(1j * np.pi * a)
                rz = 1./z
                T1 = ([E,z], [1,-a], [b], [b-a], [a, 1+a-b], [], -rz)
                T2 = ([exp(z),z], [1,a-b], [b], [a], [b-a, 1-a], [], rz)
                return T1, T2

            v = hypercomb(h, [a, b])
            return np.real(v)

        # except ValueError("Hypergeometric serie did not converge"):
        #     pass

    return hypsum1f1(a, b, z)
