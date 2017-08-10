from dipy.reconst.base import ReconstModel
import numpy as np
from scipy.optimize import least_squares
import dipy.reconst.mix as mix
import dipy.reconst.mix_fast as mix_fast
from scipy.optimize import differential_evolution
# import cvxpy as cvx
from dipy.data import get_data
import nibabel as nib
from numba import jit, cfunc, float64
from scipy.linalg import get_blas_funcs
gemm = get_blas_funcs("gemm")
from time import time
import cvxpy as cvx
from dipy.reconst.recspeed import func_mul


global overall_duration
overall_duration = 0

gamma = 2.675987 * 10 ** 8
D_intra = 0.6 * 10 ** 3
fname, fscanner = get_data('ActiveAx_synth_2d')
params = np.loadtxt(fscanner)
img = nib.load(fname)
data = img.get_data()
affine = img.affine
#bvecs = params[:, 0:3]
G = params[:, 3] / 10 ** 6  # gradient strength
#big_delta = params[:, 4]
#small_delta = params[:, 5]
D_iso = 2 * 10 ** 3

am = np.array([1.84118307861360, 5.33144196877749,
               8.53631578218074, 11.7060038949077,
               14.8635881488839, 18.0155278304879,
               21.1643671187891, 24.3113254834588,
               27.4570501848623, 30.6019229722078,
               33.7461812269726, 36.8899866873805,
               40.0334439409610, 43.1766274212415,
               46.3195966792621, 49.4623908440429,
               52.6050411092602, 55.7475709551533,
               58.8900018651876, 62.0323477967829,
               65.1746202084584, 68.3168306640438,
               71.4589869258787, 74.6010956133729,
               77.7431620631416, 80.8851921057280,
               84.0271895462953, 87.1691575709855,
               90.3110993488875, 93.4530179063458,
               96.5949155953313, 99.7367932203820,
               102.878653768715, 106.020498619541,
               109.162329055405, 112.304145672561,
               115.445950418834, 118.587744574512,
               121.729527118091, 124.871300497614,
               128.013065217171, 131.154821965250,
               134.296570328107, 137.438311926144,
               140.580047659913, 143.721775748727,
               146.863498476739, 150.005215971725,
               153.146928691331, 156.288635801966,
               159.430338769213, 162.572038308643,
               165.713732347338, 168.855423073845,
               171.997111729391, 175.138794734935,
               178.280475036977, 181.422152668422,
               184.563828222242, 187.705499575101])

#am = np.array([1.84118307861360, 5.33144196877749,
#               8.53631578218074, 11.7060038949077,
#               14.8635881488839])

am = np.array([1.84118307861360])

@jit(nopython=True, nogil=True, cache=True)
def func_bvec(bvecs, n):
    M = bvecs.shape[0]
    g_per = np.zeros((M))
    for i in range(M):
#        g_per[i] = bvecs[i, 0]**2 + bvecs[i, 1]**2 + bvecs[i, 2]**2 - \
#                   (bvecs[i, 0]*n[0] + bvecs[i, 1]*n[1] + bvecs[i, 2]*n[2])**2
        g_per[i] = 1 - \
                   (bvecs[i, 0]*n[0] + bvecs[i, 1]*n[1] + bvecs[i, 2]*n[2])**2
    return g_per


##@profile
#@jit(nopython=True, nogil=True, cache=True)
#def func_mul(x, am2, small_delta, big_delta):
#    M = am2.shape[0]
#    N = small_delta.shape[0]
#    summ = np.zeros((N, M))
#    for i in range(M):
#        am = am2[i]
#        D_intra_am = D_intra * am
#        bd = D_intra_am * big_delta
#        sd = D_intra_am * small_delta
##        num = 2 * sd - 2 + 2 * np.e**(-sd) + 2 * np.e**(-bd) - \
##                np.e**(-(bd - sd)) - np.e**(-(bd + sd))
#        num = 2 * sd - 2 + 2 * np.exp(-sd) + 2 * np.exp(-bd) - \
#                np.exp(-(bd - sd)) - np.exp(-(bd + sd))
#        denom = (D_intra ** 2) * (am ** 3) * ((x[2]) ** 2 * am - 1)
#        idenom = 1. / denom
#        summ[:, i] = num * idenom
#    return summ
#@profile
@jit(signature_or_function=float64[:](float64[:], float64[:], float64[:], float64[:]), nopython=True, nogil=True, cache=True)
def func_mul_jitted(x, am2, small_delta, big_delta):
    M = am2.shape[0]
    bd = np.zeros((small_delta.shape[0], M))
    sd = np.zeros((small_delta.shape[0], M))
    D_intra = 0.6 * 10 ** 3
    for i in range(M):
        am = am2[i]
        D_intra_am = D_intra * am
        bd[:, i] = D_intra_am * big_delta
        sd[:, i] = D_intra_am * small_delta
#    esd = np.exp(-sd)
#    ebd = np.exp(-bd)
#    num = 2 * sd - 2 + 2 * esd + 2 * ebd - \
#                ebd / esd - ebd * esd
    num = 2 * sd - 2 + 2 * np.exp(-sd) + 2 * np.exp(-bd) - \
                np.exp(-(bd - sd)) - np.exp(-(bd + sd))
#    num = all_exps(sd, bd)
    denom = (D_intra ** 2) * (am2 ** 3) * ((x[2]) ** 2 * am2 - 1)
    idenom = 1. / denom
    #summ = fun_sum(num, idenom)
    summ = num * idenom.T
    #summ_rows = np.sum(summ, axis=1)
    summ_rows = np.zeros((summ.shape[0],))
    for i in range(summ.shape[0]):
        summ_rows[i] = np.sum(summ[i])
    return summ_rows





##@cfunc("float64(float64, float64)")
#signature= "float64(float64, float64)"
#@jit(nopython=True, nogil=True, cache=True)
#def all_exps(sd, bd):
#    return 2 * sd - 2 + 2 * np.exp(-sd) + 2 * np.exp(-bd) - \
#            np.exp(-(bd - sd)) - np.exp(-(bd + sd))


#@jit(nopython=True, nogil=True, cache=True)
#def fun_sum(num, idenom):
#    summ = np.zeros((num.shape[0], num.shape[1]))
#    M = idenom.shape[0]
#    for i in range(M):
#        summ[:, i] = num[:, i] * idenom[i]
#    return summ


#@jit(nopython=True, nogil=True, cache=True)
#def S2(x2, bvals, bvecs):
#    x2_0 = x2[0]
#    x2_1 = x2[1]
#    x2_2 = x2[2]
#    sinT = np.sin(x2_0)
#    cosT = np.cos(x2_0)
#    sinP = np.sin(x2_1)
#    cosP = np.cos(x2_1)
#    n = np.array([cosP * sinT, sinP * sinT, cosT])
#    # zeppelin
#    yhat_zeppelin = bvals * ((D_intra - (D_intra * (1 - x2_2))) *
#                        (np.dot(bvecs, n) ** 2) + (D_intra *
#                                                             (1 - x2_2)))
#    return yhat_zeppelin


class ActiveAxModel(ReconstModel):

    def __init__(self, gtab, fit_method='MIX'):

        self.gtab = gtab
        self.big_delta = gtab.big_delta
        self.small_delta = gtab.small_delta
        self.gamma = gamma
        self.G = G
        D_iso = 2 * 10 ** 3
        self.yhat_ball = D_iso * self.gtab.bvals
        self.summ = np.zeros((self.small_delta.shape[0], am.shape[0]))
#        self.bvec_norm = func_norm(gtab.bvecs)

    @profile
    def S1(self, x1):
        big_delta = self.big_delta
        small_delta = self.small_delta
        sinT = np.sin(x1[0])
        cosT = np.cos(x1[0])
        sinP = np.sin(x1[1])
        cosP = np.cos(x1[1])
        n = np.array([cosP * sinT, sinP * sinT, cosT])
        # Cylinder
        L = self.gtab.bvals * D_intra
        L1 = L * np.dot(self.gtab.bvecs, n) ** 2
        am2 = (am / x1[2]) ** 2
        t1 = time()

        summ_rows = np.zeros(small_delta.shape[0])
        func_mul(x1, am2, small_delta, big_delta, summ_rows)
        t2 = time()
        duration = t2 - t1
        global overall_duration
        overall_duration += duration
#        summ_rows = np.sum(summ, axis=1)
        g_per = func_bvec(self.gtab.bvecs, n)
        L2 = 2 * (g_per * gamma ** 2) * summ_rows * G ** 2
        yhat_cylinder = L1 + L2
        return yhat_cylinder

    def S4(self):
        # dot
        yhat_dot = np.zeros(self.gtab.bvals.shape)
        return yhat_dot

#    @jit(nogil=True, cache=True)
#    def func_mul(self, x, am2):
#        M = am2.shape[0]
#        big_delta = self.big_delta
#        small_delta = self.small_delta
##        summ = np.zeros((self.small_delta.shape[0], M))
#        for i in range(M):
#            am = am2[i]
#            D_intra_am = D_intra * am
#            bd = D_intra_am * big_delta
#            sd = D_intra_am * small_delta
#            num = 2 * sd - 2 + 2 * np.exp(-sd) + 2 * np.exp(-bd) - \
#                  np.exp(-(bd - sd)) - np.exp(-(bd + sd))
#            denom = (D_intra ** 2) * (am ** 3) * ((x[2]) ** 2 * am - 1)
#            idenom = 1. / denom
#            self.summ[:, i] = num * idenom
#        return self.summ
#
##    @jit(nogil=True, cache=True)
##    def func_norm(self):
##        bvec_norm = np.zeros(self.gtab.bvecs.shape[0])
##        for i in range(self.gtab.bvecs.shape[0]):
##            bvec_norm[i] = self.gtab.bvecs[i, 0]**2 + \
##                     self.gtab.bvecs[i, 1]**2 + self.gtab.bvecs[i, 2]**2
##        return bvec_norm
#
#    @jit(nogil=True, cache=True)
#    def func_bvec(self, n):
#        M = self.gtab.bvecs.shape[0]
#        g_per = np.zeros((M))
#        bvecs = self.gtab.bvecs
#        for i in range(M):
##            g_per[i] = bvecs[i, 0]**2 + bvecs[i, 1]**2 + bvecs[i, 2]**2 - \
##                       (bvecs[i, 0]*n[0] + bvecs[i, 1]*n[1] + bvecs[i, 2]*n[2])**2
#            g_per[i] = 1 - \
#                       (bvecs[i, 0]*n[0] + bvecs[i, 1]*n[1] + bvecs[i, 2]*n[2])**2
#
#        return g_per

    def x_to_xs(self, x):
        x1 = x[0:3]
        x2 = np.zeros((3))
        x2[0:2] = x[0:2]
        x2[2] = x[3]
        return x1, x2


    def xs_to_x(self):
        pass

    @jit(nogil=True, cache=True)
    def S2(self, x2):
        x2_0 = x2[0]
        x2_1 = x2[1]
        x2_2 = x2[2]
        sinT = np.sin(x2_0)
        cosT = np.cos(x2_0)
        sinP = np.sin(x2_1)
        cosP = np.cos(x2_1)
        n = np.array([cosP * sinT, sinP * sinT, cosT])
        # zeppelin
        yhat_zeppelin = self.gtab.bvals * ((D_intra - (D_intra * (1 - x2_2))) *
                        (np.dot(self.gtab.bvecs, n) ** 2) + (D_intra *
                                                             (1 - x2_2)))
        return yhat_zeppelin


    def S2_new(self, x_fe):
        x, fe = self.x_fe_to_x_and_fe(x_fe)
        fe0 = fe[0]
        x_0 = x[0]
        x_1 = x[1]
        sinT = np.sin(x_0)
        cosT = np.cos(x_0)
        sinP = np.sin(x_1)
        cosP = np.cos(x_1)
        n = np.array([cosP * sinT, sinP * sinT, cosT])
        # zeppelin
        v = fe0/(fe0 + fe[1])
        yhat_zeppelin = self.gtab.bvals * ((D_intra - (D_intra * (1 - v))) *
                                  (np.dot(self.gtab.bvecs, n) ** 2) + (D_intra *
                                                                  (1 - v)))
        return yhat_zeppelin

    def S3(self):
        # ball
        yhat_ball = D_iso * self.gtab.bvals
        return yhat_ball

#    @profile
    def Phi(self, x):
        x1, x2 = self.x_to_xs(x)
        yhat_zeppelin = self.S2(x2)
#        yhat_zeppelin = S2(x2, self.gtab.bvals, self.gtab.bvecs)
        yhat_cylinder = self.S1(x1)
        phi = np.vstack([yhat_cylinder, yhat_zeppelin, self.S3(),
                         self.S4()]).T
        phi = np.ascontiguousarray(phi)
        return np.exp(-phi)

#    @profile
    def Phi2(self, x_fe):
        x, fe = self.x_fe_to_x_and_fe(x_fe)
        x1 = x[0:3]
        yhat_zeppelin = self.S2_new(x_fe)
        yhat_cylinder = self.S1(x1)
        phi = np.vstack([yhat_cylinder, yhat_zeppelin, self.S3(),
                         self.S4()]).T
        phi = np.ascontiguousarray(phi)
        return np.exp(-phi)

    def bounds(self, x):
        bound = ([0.01, 0.01,  0.01, 0.01, 0.01, 0.1, 0.01], [0.9,  0.9,  0.9,
                 np.pi, np.pi, 11, 0.9])
        return bound

    def bounds_x_fe(self, x_fe):
        pass

    def x_fe_to_x_and_fe(self, x_fe):
        fe = np.zeros((1, 4))
        fe = np.squeeze(fe)
        fe[0:3] = x_fe[0:3]
        fe[3] = x_fe[6]
        x = x_fe[3:6]
        return x, fe

    def x_and_fe_to_x_fe(self, x, fe):
        x_fe = np.zeros([7])
        fe = np.squeeze(fe)
        x_fe[:3] = fe[:3]
        x_fe[3:6] = x[:3]
        x_fe[6] = fe[3]
        return x_fe

#    def estimate_signal(self, x_fe):
#        x, fe = self.x_fe_to_x_and_fe(x_fe)
#        x1, x2 = self.x_to_xs(x)
#        S = fe[0] * self.S1(x1) + fe[1] * self.S2(x2) + fe[2] * self.S3() \
#              + fe[3] * self.S4()
#        return S

#    @profile
    def stoc_search_cost(self, x, signal):

        """
        Aax_exvivo_nlin

        Cost function for genetic algorithm

        Parameters
        ----------
        x : array
            x.shape = 4x1
            x(0) theta (radian)
            x(1) phi (radian)
            x(2) R (micrometers)
            x(3) v=f1/(f1+f2) (0.1 - 0.8)

        bvals
        bvecs
        G: gradient strength
        small_delta
        big_delta
        gamma: gyromagnetic ratio (2.675987 * 10 ** 8 )
        D_intra= intrinsic free diffusivity (0.6 * 10 ** 3 mircometer^2/sec)
        D_iso= isotropic diffusivity, (2 * 10 ** 3 mircometer^2/sec)

        Returns
        -------
        (signal -  S)^T(signal -  S)

        Notes
        --------
        cost function for genetic algorithm:

            .. math::

                (signal -  S)^T(signal -  S)
        """
        phi = self.Phi(x)
        error_one = mix.activeax_cost_one(phi, signal)
        return error_one

#    @profile
    def nls_cost(self, x_fe, signal):

        """
    Aax_exvivo_eval

    cost function for the least square problem

    Parameters
    ----------
    x_fe : array
        x_fe(0) x_fe(1) x_fe(2)  are f1 f2 f3
        x_fe(3) theta
        x_fe(4) phi
        x_fe(5) R
        x_fe(6) as f4

    signal_param : array
        signal_param.shape = number of data points x 7

        signal_param = np.hstack([signal[:, None], bvals[:, None], bvecs,
                        G[:, None], small_delta[:, None], big_delta[:, None]])

    Returns
    -------
    sum{(signal -  phi*fe)^2}

    Notes
    --------
    cost function for the least square problem

    .. math::

        sum{(signal -  phi*fe)^2}
        """

        x, fe = self.x_fe_to_x_and_fe(x_fe)
        phi = self.Phi2(x_fe)
        return np.sum((np.squeeze(np.dot(phi, fe)) - signal) ** 2)


    def estimate_f(signal, phi):

        """
    Linear parameters fit using cvx

    Parameters
    ----------
    phi : array
        phi.shape = number of data points x 4
    signal : array
        signal.shape = number of data points x 1

    Returns
    -------
    f1, f2, f3, f4 (volume fractions)
    f1 = fe[0]
    f2 = fe[1]
    f3 = fe[2]
    f4 = fe[3]

    Notes
    --------
    cost function for genetic algorithm:

    .. math::

        minimize(norm((signal)- (phi*fe)))
        """

        # Create four scalar optimization variables.
        fe = cvx.Variable(4)
        # Create four constraints.
        constraints = [cvx.sum_entries(fe) == 1,
                       fe[0] >= 0,
                       fe[1] >= 0,
                       fe[2] >= 0,
                       fe[3] >= 0]

        # Form objective.
        obj = cvx.Minimize(cvx.sum_entries(cvx.square(phi * fe - signal[:, None])))

        # Form and solve problem.
        prob = cvx.Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.

        return np.array(fe.value)


    def fit(self, data, mask=None):
        bounds = [(0.01, np.pi), (0.01, np.pi), (0.1, 11), (0.1, 0.8)]
        res_one = differential_evolution(self.stoc_search_cost, bounds,
                                         args=(data,))
        x = res_one.x
        phi = self.Phi(x)
#        fe = mix_fast.estimate_f(np.array(data), phi)
        fe = self.estimate_f(np.array(data), phi)
        x_fe = self.x_and_fe_to_x_fe(x, fe)
        bounds = ([0.01, 0.01,  0.01, 0.01, 0.01, 0.1, 0.01], [0.9,  0.9,  0.9,
                  np.pi, np.pi, 11, 0.9])
        res = least_squares(self.nls_cost, x_fe, bounds=(bounds),
                            args=(data,))
        result = res.x
        return result
