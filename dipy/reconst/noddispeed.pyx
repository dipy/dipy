#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
cimport cython
import numpy as np
cimport numpy as cnp
from scipy.special.cython_special cimport erf
from libc.math cimport sqrt, sin, cos, pow, pi, exp

"""
Following is the Cythonized Module for the Legendre Polynomial upto the power
of 14, as required by the NODDIx model.
Note: This is an alternative method that statically performs the numerical
computations as opposed to the scipy module.
"""

cdef inline double p0(double x) nogil:
    return 1

cdef inline double p1(double x) nogil:
    return x

cdef inline double p2(double x) nogil:
    return 1.5 * pow(x, 2) - 0.5

cdef inline double p3(double x) nogil:
    return 2.5*pow(x, 3) - 1.5*x

cdef inline double p4(double x) nogil:
    return 4.37500000e+00*pow(x, 4) + 4.85722573e-16*pow(x, 3) - 3.75000000e+00*pow(x,2)\
        + 2.42861287e-16*x + 3.75000000e-01

cdef inline double p5(double x) nogil:
    return 7.87500000e+00*pow(x, 5) - 8.75000000e+00*pow(x, 3) - 4.37150316e-16*pow(x,2)\
        +  1.87500000e+00*x

cdef inline double p6(double x) nogil:
    return 1.44375000e+01*pow(x,6) - 1.96875000e+01*pow(x,4) \
        + 1.60288449e-15*pow(x,3) + 6.56250000e+00*pow(x, 2) - 3.12500000e-01

cdef inline double p7(double x) nogil:
    return 2.68125000e+01*pow(x,7) + 5.95357097e-15*pow(x,6) -4.33125000e+01*pow(x,5)\
        + 2.97678548e-15*pow(x,4) + 1.96875000e+01*pow(x,3) - 1.11629456e-15*pow(x,2)\
        - 2.18750000e+00*x

cdef inline double p8(double x) nogil:
    return 5.02734375e+01*pow(x, 8) - 5.58147278e-15*pow(x, 7) - 9.38437500e+01*pow(x, 6)\
        - 2.23258911e-14*pow(x, 5) + 5.41406250e+01*pow(x, 4) - 5.58147278e-15*pow(x, 3)\
        - 9.84375000e+00*pow(x, 2) - 4.36052561e-17*pow(x, 1) + 2.73437500e-01

cdef inline double p9(double x) nogil:
    return 9.49609375e+01*pow(x, 9) - 2.10855639e-14*pow(x, 8) - 2.01093750e+02*pow(x, 7)\
        + 6.32566916e-14*pow(x, 6) + 1.40765625e+02*pow(x, 5) + 1.58141729e-14*pow(x, 4)\
        - 3.60937500e+01*pow(x, 3) - 1.97677161e-15*pow(x, 2) + 2.46093750e+00*x

cdef inline double p10(double x) nogil:
    return 1.80425781e+02*pow(x, 10) - 4.00625713e-14*pow(x, 9) - 4.27324219e+02*pow(x, 8)\
        - 2.80437999e-13*pow(x, 7) + 3.51914063e+02*pow(x, 6) - 2.80437999e-13*pow(x, 5)\
        - 1.17304687e+02*pow(x, 4) - 1.00156428e-14*pow(x, 3) + 1.35351562e+01*pow(x, 2)\
        + 5.47730467e-16*x - 2.46093750e-01

cdef inline double p11(double x) nogil:
    return 3.44449219e+02*pow(x, 11) - 7.64830907e-14*pow(x, 10) - 9.02128906e+02*pow(x, 9)\
        - 6.88347816e-13*pow(x, 8)  + 8.54648438e+02*pow(x, 7) - 3.05932363e-13*pow(x, 6)\
        - 3.51914062e+02*pow(x, 5)  + 2.86811590e-14*pow(x, 4) + 5.86523438e+01*pow(x, 3)\
        - 2.70703125e+00*x

cdef inline double p12(double x) nogil:
    return 6.60194336e+02*pow(x, 12) - 2.19888886e-13*pow(x, 11) - 1.89447070e+03*pow(x, 10)\
        - 2.93185181e-13*pow(x, 9) - 2.02979004e+03*pow(x, 8) - 1.02614813e-12*pow(x, 7) - 9.97089844e+02*pow(x, 6) - 3.66481476e-13*pow(x, 5) + 2.19946289e+02*pow(x, 4)\
        - 6.87152768e-15*pow(x, 3) - 1.75957031e+01*pow(x, 2) - 7.15784133e-16*x\
        + 2.25585938e-01

cdef inline double p13(double x) nogil:
    return 1.26960449e+03*pow(x, 13) - 2.81908828e-13*pow(x, 12) - 3.96116602e+03*pow(x, 11)\
        - 5.07435890e-12*pow(x, 10) + 4.73617676e+03*pow(x, 9) +2.53717945e-12*pow(x, 8) - 2.70638672e+03*pow(x, 7) + 2.81908828e-13*pow(x, 6) + 7.47817383e+02*pow(x, 5)\
        - 1.76193017e-13*pow(x, 4) - 8.79785156e+01*pow(x, 3) - 5.50603179e-15*pow(x, 2)\
        + 2.93261719e+00*x

cdef inline double p14(double x) nogil:
    return 2.44852295e+03*pow(x, 14) - 1.35920328e-12*pow(x, 13) - 8.25242920e+03*pow(x, 12)\
        - 4.34945049e-12*pow(x, 11) + 1.08932065e+04*pow(x, 10) - 1.30483515e-11*pow(x, 9) - 7.10426514e+03*pow(x, 8) - 9.24258229e-12*pow(x, 7) + 2.36808838e+03*pow(x, 6)\
        - 1.15532279e-12*pow(x, 5) - 3.73908691e+02*pow(x, 4) - 4.24751024e-14*pow(x, 3) + 2.19946289e+01*pow(x, 2) - 1.32734695e-16*x - 2.09472656e-01

cdef inline double legendre_eval(cnp.npy_intp n, double x) nogil:

    if n == 0:
        return p0(x)
    elif n == 1:
        return p1(x)
    elif n == 2:
        return p2(x)
    elif n == 3:
        return p3(x)
    elif n == 4:
        return p4(x)
    elif n == 5:
        return p5(x)
    elif n == 6:
        return p6(x)
    elif n == 7:
        return p7(x)
    elif n == 8:
        return p8(x)
    elif n == 9:
        return p10(x)
    elif n == 10:
        return p10(x)
    elif n == 11:
        return p11(x)
    elif n == 12:
        return p12(x)
    elif n == 13:
        return p13(x)
    elif n == 14:
        return p14(x)


def legendre(n, x):
    return legendre_eval(n, x)


def legendre_matrix(cnp.npy_intp n, double [:] x, double[:] out):
    cdef cnp.npy_intp i
    cdef cnp.npy_intp shape = x.shape[0]

    with nogil:
        for i in range(shape):
            out[i] = legendre_eval(n, x[i])


#def error_function(double [:] x, double [:] out):
#    cdef cnp.npy_intp i
#    cdef cnp.npy_intp shape = x.shape[0]
#
#    with nogil:
#        for i in range(shape):
#            out[i] = erf(x[i])


def legendre_gauss_integral(double[:] x_vec, cnp.npy_intp n):
    # creating the 2D array of zeros, modified by both if and else
    cdef:
        double[:, :] I = np.zeros((x_vec.shape[0], n + 1))
        double[:, :] L = np.zeros((x_vec.shape[0], n + 1))
        cnp.npy_intp cnt
        double dx, emx, sqrtx, x

    for cnt in range(x_vec.shape[0]):
        x = x_vec[cnt]
        if x > 0.05:
            sqrtx = sqrt(x)
            I[cnt, 0] = sqrt(pi) * erf(x) / sqrtx
            dx = 1 / x
            emx = -exp(-x)
            for i in range(2, n + 2):
                I[cnt, i - 1] = emx + (i - 1.5) * I[cnt, i - 2]
                I[cnt, i - 1] = I[cnt, i - 1] * dx

            L[cnt, 0] = I[cnt, 0]
            L[cnt, 1] = -0.5 * I[cnt, 0] + 1.5 * I[cnt, 1]
            L[cnt, 2] = 0.375 * I[cnt, 0] - 3.75 * I[cnt, 1] + 4.375 * I[cnt, 2]
            L[cnt, 3] = -0.3125 * I[cnt, 0] + 6.5625 * I[cnt, 1] - 19.6875 * I[cnt, 2] + 14.4375 * I[cnt, 3]
            L[cnt, 4] = 0.2734375 * I[cnt, 0] - 9.84375 * I[cnt, 1] + 54.140625 * I[cnt, 2] - 93.84375 * I[cnt, 3] + 50.2734375 * I[cnt, 4]
            L[cnt, 5] = -(63. / 256) * I[cnt, 0] + (3465. / 256) * I[cnt, 1] - (30030. / 256) * I[cnt, 2] + (90090. / 256) * I[cnt, 3] - (109395. / 256) * I[cnt, 4] + (46189. / 256) * I[cnt, 5]
            L[cnt, 6] = (231. / 1024) * I[cnt, 0] - (18018. / 1024) * I[cnt, 1] + (225225. / 1024) * I[cnt, 2] - (1021020. / 1024) * I[cnt, 3] + (2078505. / 1024) * I[cnt, 4] - (1939938. / 1024) * I[cnt, 5] + (676039. / 1024) * I[cnt, 6]

        elif x <= 0.05:
            x2 = pow(x, 2)
            x3 = x2 * x
            x4 = x3 * x
            x5 = x4 * x
            x6 = x5 * x
            L[cnt, 0] = 2 - 2 * x / 3 + x2 / 5 - x3 / 21 + x4 / 108
            L[cnt, 1] = -4 * x / 15 + 4 * x2 / 35 - 2 * x3 / 63 + 2 * x4 / 297
            L[cnt, 2] = 8 * x2 / 315 - 8 * x3 / 693 + 4 * x4 / 1287
            L[cnt, 3] = -16 * x3 / 9009 + 16 * x4 / 19305
            L[cnt, 4] = 32 * x4 / 328185
            L[cnt, 5] = -64 * x5 / 14549535
            L[cnt, 6] = 128 * x6 / 760543875
    return L

