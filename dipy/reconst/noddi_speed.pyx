#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
cimport cython
from cython cimport floating
import numpy as np
cimport numpy as cnp
from scipy.special.cython_special cimport erf, erfi
from libc.math cimport sqrt, sin, cos, pow, pi, exp, log

"""
Following is the Cythonized Module for the Legendre Polynomial upto the power
of 14, as required by the NODDIx model.
Note: This is an alternative method that statically performs the numerical
computations as opposed to the scipy module.
"""
cdef inline double legendre_eval(cnp.npy_intp n, double x) nogil:
    if n == 0:
         return 1
    elif n == 1:
        return x
    elif n == 2:
        return 1.5 * pow(x, 2) - 0.5
    elif n == 3:
        return 2.5*pow(x, 3) - 1.5*x
    elif n == 4:
        return 4.37500000e+00*pow(x, 4) + 4.85722573e-16*pow(x, 3) - 3.75000000e+00*pow(x,2)\
        + 2.42861287e-16*x + 3.75000000e-01
    elif n == 5:
        return 7.87500000e+00*pow(x, 5) - 8.75000000e+00*pow(x, 3) - 4.37150316e-16*pow(x,2)\
        +  1.87500000e+00*x
    elif n == 6:
        return 1.44375000e+01*pow(x,6) - 1.96875000e+01*pow(x,4)\
        + 1.60288449e-15*pow(x,3) + 6.56250000e+00*pow(x, 2) - 3.12500000e-01
    elif n == 7:
        return 2.68125000e+01*pow(x,7) + 5.95357097e-15*pow(x,6) -4.33125000e+01*pow(x,5)\
        + 2.97678548e-15*pow(x,4) + 1.96875000e+01*pow(x,3) - 1.11629456e-15*pow(x,2)\
        - 2.18750000e+00*x
    elif n == 8:
        return 5.02734375e+01*pow(x, 8) - 5.58147278e-15*pow(x, 7) - 9.38437500e+01*pow(x, 6)\
        - 2.23258911e-14*pow(x, 5) + 5.41406250e+01*pow(x, 4) - 5.58147278e-15*pow(x, 3)\
        - 9.84375000e+00*pow(x, 2) - 4.36052561e-17*pow(x, 1) + 2.73437500e-01
    elif n == 9:
        return 9.49609375e+01*pow(x, 9) - 2.10855639e-14*pow(x, 8) - 2.01093750e+02*pow(x, 7)\
        + 6.32566916e-14*pow(x, 6) + 1.40765625e+02*pow(x, 5) + 1.58141729e-14*pow(x, 4)\
        - 3.60937500e+01*pow(x, 3) - 1.97677161e-15*pow(x, 2) + 2.46093750e+00*x
    elif n == 10:
        return 1.80425781e+02*pow(x, 10) - 4.00625713e-14*pow(x, 9) - 4.27324219e+02*pow(x, 8)\
        - 2.80437999e-13*pow(x, 7) + 3.51914063e+02*pow(x, 6) - 2.80437999e-13*pow(x, 5)\
        - 1.17304687e+02*pow(x, 4) - 1.00156428e-14*pow(x, 3) + 1.35351562e+01*pow(x, 2)\
        + 5.47730467e-16*x - 2.46093750e-01
    elif n == 11:
        return 3.44449219e+02*pow(x, 11) - 7.64830907e-14*pow(x, 10) - 9.02128906e+02*pow(x, 9)\
        - 6.88347816e-13*pow(x, 8)  + 8.54648438e+02*pow(x, 7) - 3.05932363e-13*pow(x, 6)\
        - 3.51914062e+02*pow(x, 5)  + 2.86811590e-14*pow(x, 4) + 5.86523438e+01*pow(x, 3)\
        - 2.70703125e+00*x
    elif n == 12:
        return 6.60194336e+02*pow(x, 12) - 2.19888886e-13*pow(x, 11) - 1.89447070e+03*pow(x, 10)\
        - 2.93185181e-13*pow(x, 9) - 2.02979004e+03*pow(x, 8) - 1.02614813e-12*pow(x, 7) - 9.97089844e+02*pow(x, 6) - 3.66481476e-13*pow(x, 5) + 2.19946289e+02*pow(x, 4)\
        - 6.87152768e-15*pow(x, 3) - 1.75957031e+01*pow(x, 2) - 7.15784133e-16*x\
        + 2.25585938e-01
    elif n == 13:
        return 1.26960449e+03*pow(x, 13) - 2.81908828e-13*pow(x, 12) - 3.96116602e+03*pow(x, 11)\
        - 5.07435890e-12*pow(x, 10) + 4.73617676e+03*pow(x, 9) +2.53717945e-12*pow(x, 8) - 2.70638672e+03*pow(x, 7) + 2.81908828e-13*pow(x, 6) + 7.47817383e+02*pow(x, 5)\
        - 1.76193017e-13*pow(x, 4) - 8.79785156e+01*pow(x, 3) - 5.50603179e-15*pow(x, 2)\
        + 2.93261719e+00*x
    elif n == 14:
        return 2.44852295e+03*pow(x, 14) - 1.35920328e-12*pow(x, 13) - 8.25242920e+03*pow(x, 12)\
        - 4.34945049e-12*pow(x, 11) + 1.08932065e+04*pow(x, 10) - 1.30483515e-11*pow(x, 9) - 7.10426514e+03*pow(x, 8) - 9.24258229e-12*pow(x, 7) + 2.36808838e+03*pow(x, 6)\
        - 1.15532279e-12*pow(x, 5) - 3.73908691e+02*pow(x, 4) - 4.24751024e-14*pow(x, 3) + 2.19946289e+01*pow(x, 2) - 1.32734695e-16*x - 2.09472656e-01


def legendre_gauss_integral(double[:] x_vec, cnp.npy_intp n):
    # creating the 2D array of zeros, modified by both if and else
    cdef:
        cnp.ndarray[cnp.float64_t, ndim=2] I = np.empty((x_vec.shape[0], n + 1))
        cnp.ndarray[cnp.float64_t, ndim=2] L = np.empty((x_vec.shape[0], n + 1))

        cnp.npy_intp cnt, i
        double dx, emx, sqrtx, x
        double x2, x3, x4, x5, x6

    with nogil:
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
                L[cnt, 0] = 2. - 2. * x / 3. + x2 / 5. - x3 / 21. + x4 / 108.
                L[cnt, 1] = -4. * x / 15. + 4. * x2 / 35. - 2. * x3 / 63. + 2. * x4 / 297.
                L[cnt, 2] = 8. * x2 / 315. - 8. * x3 / 693. + 4. * x4 / 1287.
                L[cnt, 3] = -16. * x3 / 9009. + 16. * x4 / 19305.
                L[cnt, 4] = 32. * x4 / 328185.
                L[cnt, 5] = -64. * x5 / 14549535.
                L[cnt, 6] = 128. * x6 / 760543875.
    return L

def watson_sh_coeff(double k):
    cdef:
        double[:] C = np.empty((7))
        double sk, sk2, sk3, sk4, sk5, sk6
        double k2, k3, k4, k5, k6
        double erfik, ierfik, dawsonk, ek
        double lnkd, lnkd1, lnkd2, lnkd3, lnkd4, lnkd5, lnkd6

    C[0] = 2 * sqrt(pi)
    # Precompute the special function values
    sk = sqrt(k)
    sk2 = sk * k
    sk3 = sk2 * k
    sk4 = sk3 * k
    sk5 = sk4 * k
    sk6 = sk5 * k
    k2 = k ** 2
    k3 = k2 * k
    k4 = k3 * k
    k5 = k4 * k
    k6 = k5 * k

    erfik = erfi(sk)
    ierfik = 1 / erfik
    ek = exp(k)
    dawsonk = 0.5 * sqrt(pi) * erfik / ek
    with nogil:
        if k > 0.1:
        # for large enough kappa
            C[1] = 3. * sk - (3. + 2. * k) * dawsonk
            C[1] = sqrt(5) * C[1] * ek
            C[1] = C[1] * ierfik / k
            C[2] = (105. + 60. * k + 12. * k2) * dawsonk
            C[2] = C[2] - 105. * sk + 10. * sk2
            C[2] = .375 * C[2] * ek / k2
            C[2] = C[2] * ierfik
            C[3] = -3465. - 1890. * k - 420. * k2 - 40. * k3
            C[3] = C[3] * dawsonk
            C[3] = C[3] + 3465. * sk - 420. * sk2 + 84. * sk3
            C[3] = C[3] * sqrt(13 * pi) / 64 / k3
            C[3] = C[3] / dawsonk
            C[4] = 675675. + 360360. * k + 83160. * k2 + 10080. * k3 + 560. * k4
            C[4] = C[4] * dawsonk
            C[4] = C[4] - 675675. * sk + 90090. * sk2 - 23100. * sk3 + 744. * sk4
            C[4] = sqrt(17) * C[4] * ek
            C[4] = C[4] / 512. / k4
            C[4] = C[4] * ierfik
            C[5] = -43648605. - 22972950. * k - 5405400. * k2 - 720720. * k3 - 55440. * k4 - 2016. * k5
            C[5] = C[5] * dawsonk
            C[5] = C[5] + 43648605. * sk - 6126120. * sk2 + 1729728. * sk3 - 82368. * sk4 + 5104. * sk5
            C[5] = sqrt(21 * pi) * C[5] / 4096. / k5
            C[5] = C[5] / dawsonk
            C[6] = 7027425405. + 3666482820. * k + 872972100 * k2 + 122522400. * k3 + 10810800. * k4 + 576576. * k5 + 14784. * k6
            C[6] = C[6] * dawsonk
            C[6] = C[6] - 7027425405. * sk + 1018467450. * sk2 - 302630328. * sk3 + 17153136. * sk4 - 1553552. * sk5 + 25376. * sk6
            C[6] = 5 * C[6] * ek
            C[6] = C[6] / 16384. / k6
            C[6] = C[6] * ierfik

        elif k > 30:
        # for very large kappa
            lnkd = log(k) - log(30)
            lnkd2 = lnkd * lnkd
            lnkd3 = lnkd2 * lnkd
            lnkd4 = lnkd3 * lnkd
            lnkd5 = lnkd4 * lnkd
            lnkd6 = lnkd5 * lnkd
            C[1] = 7.52308 + 0.411538 * lnkd - 0.214588 * lnkd2 \
                    + 0.0784091 * lnkd3 - 0.023981 * lnkd4 + 0.00731537 * lnkd5 \
                    - 0.0026467 * lnkd6
            C[2] = 8.93718 + 1.62147 * lnkd - 0.733421 * lnkd2 \
                    + 0.191568 * lnkd3 - 0.0202906 * lnkd4 - 0.00779095 * lnkd5 \
                    + 0.00574847*lnkd6
            C[3] = 8.87905 + 3.35689 * lnkd - 1.15935 * lnkd2 \
                    + 0.0673053 * lnkd3 + 0.121857 * lnkd4 - 0.066642 * lnkd5 \
                    + 0.0180215 * lnkd6
            C[4] = 7.84352 + 5.03178 * lnkd - 1.0193 * lnkd2 \
                    - 0.426362 * lnkd3 + 0.328816 * lnkd4 - 0.0688176 * lnkd5 \
                    - 0.0229398 * lnkd6
            C[5] = 6.30113 + 6.09914 * lnkd - 0.16088 * lnkd2 \
                    - 1.05578 * lnkd3 + 0.338069 * lnkd4 + 0.0937157 * lnkd5 \
                    - 0.106935 * lnkd6
            C[6] = 4.65678 + 6.30069 * lnkd + 1.13754 * lnkd2 \
                    - 1.38393 * lnkd3 - 0.0134758 * lnkd4 + 0.331686 * lnkd5 \
                    - 0.105954 * lnkd6

        elif k <= 0.1:
        # for small kappa
            C[1] = 4 / 3 * k + 8 / 63 * k2
            C[1] = C[1] * sqrt(pi / 5)
            C[2] = 8 / 21 * k2 + 32 / 693 * k3
            C[2] = C[2] * (sqrt(pi) * 0.2)
            C[3] = 16 / 693 * k3 + 32 / 10395 * k4
            C[3] = C[3] * sqrt(pi / 13)
            C[4] = 32 / 19305 * k4
            C[4] = C[4] * sqrt(pi / 17)
            C[5] = 64 * sqrt(pi / 21) * k5 / 692835
            C[6] = 128 * sqrt(pi) * k6 / 152108775
    return C

    def synthMeasSHFor(double[:] cosTheta, double[:, :] shMatrix):
        cdef:
            cnp.npy_intp i, j
            cnp.npy_intp shape = cosTheta.shape[0]
            double shMatrix1
            double[:] out

        with nogil:
            for i in range(7):
                shMatrix1 = sqrt((i + 1 - .75) / pi)
                for j in range(shape):
                    out[j] = legendre_eval(2 * (j + 1) - 2, cosTheta[j])
                shMatrix[:, i] = shMatrix1 * out

