from dipy.reconst.base import ReconstModel


class ActiveAxModel(ReconstModel):

    def __init__(self, gtab, fit_method='MIX'):

        self.gtab = gtab
        self.delta = gtab.delta
        self.gamma = gamma

    def x_to_xs(self, x):
        x1 = x[0:3]
        x2 = x[3]
        return x1, x2

    def xs_to_x(self):
        pass

    def S1(self, x1):
        x = x1
        sinT = np.sin(x[0])
        cosT = np.cos(x[0])
        sinP = np.sin(x[1])
        cosP = np.cos(x[1])
        n = np.array([cosP * sinT, sinP * sinT, cosT])

        # Cylinder
        L = bvals * D_intra
        print(L)
        print(bvecs.shape)
        print(n.shape)
        L1 = L * np.dot(bvecs, n) ** 2
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

    am2 = (am / x[2]) ** 2

    summ = np.zeros((len(bvals), len(am)))

    for i in range(len(am)):
        num = (2 * D_intra * am2[i] * small_delta) - 2 + \
              (2 * np.exp(-(D_intra * am2[i] * small_delta))) + \
              (2 * np.exp(-(D_intra * am2[i] * big_delta))) - \
              (np.exp(-(D_intra * am2[i] * (big_delta - small_delta)))) - \
              (np.exp(-(D_intra * am2[i] * (big_delta + small_delta))))

        denom = (D_intra ** 2) * (am2[i] ** 3) * ((x[2]) ** 2 * am2[i] - 1)
        summ[:, i] = num / denom

    summ_rows = np.sum(summ, axis=1)
    g_per = np.zeros(bvals.shape)

    for i in range(len(bvecs)):
        g_per[i] = np.dot(bvecs[i, :], bvecs[i, :]) - \
                   np.dot(bvecs[i, :], n) ** 2

    L2 = 2 * (g_per * gamma ** 2) * summ_rows * G ** 2

    yhat_cylinder = L1 + L2

    return yhat_cylinder

    def S2(x2):
        x = x2
        # zeppelin
        yhat_zeppelin = self.gtab.bvals * ((D_intra - (D_intra * (1 - x[3]))) *
                (np.dot(bvecs, n) ** 2) + (D_intra * (1 - x[3])))
        return yhat_zeppelin

    def S3(x3):
        # ball
        yhat_ball = (D_iso * self.gtab.bvals)
        return yhat_ball

    def S4(x4):
        # dot
        yhat_dot = 1
    return yhat_dot

    def Phi(x):
        pass

    def bounds(x):
        pass

    def bounds_x_fe(x_fe):
        pass

    def x_fe_to_x_and_fe(x_fe):
        pass

    def x_and_fe_to_x_fe(x, fe):
        pass

    def estimate_signal(x_fe):

        f1 * S1(x1) + f2 * S2(x2)
        pass


    def fit(data, mask=None):

        pass


    def activax_exvivo_compartments(x, bvals, bvecs, G, small_delta, big_delta,
                                gamma=gamma, D_intra=0.6 * 10 ** 3,
                                D_iso=2 * 10 ** 3, debug=False):

    """
    Aax_exvivo_nlin

    Parameters
    ----------
    x : array
        x.shape = 4x1
        x(0) theta (radian)
        x(1) phi (radian)
        x(2) R (micrometers)
        x(3) v=f1/(f1+f2) (ranges from 0.1 to 0.8)

    bvals : array
        bvals.shape = number of data points x 1
    bvecs : array
    G: gradient strength
    small_delta : array
    big_delta : array
    gamma: gyromagnetic ratio (2.675987 * 10 ** 8 )
    D_intra= intrinsic free diffusivity (0.6 * 10 ** 3 mircometer^2/sec)
    D_iso= isotropic diffusivity, (2 * 10 ** 3 mircometer^2/sec)

    Returns
    -------
    yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot

    Notes
    --------
    The estimated dMRI normalized signal SActiveAx is assumed to be
    coming from the following four compartments:

    .. math::

        S_hat_ActiveAx = {f1}{exp(-yhat_cylinder)}+
                    {f2}{exp(-yhat_zeppelin)}+
                    {f3}{exp(-yhat_ball)}+
                    {f4}{exp(-yhat_dot)}

        where d_perp=D_intra*(1-v)

    """

    sinT = np.sin(x[0])
    cosT = np.cos(x[0])
    sinP = np.sin(x[1])
    cosP = np.cos(x[1])
    n = np.array([cosP * sinT, sinP * sinT, cosT])

    # Cylinder
    L = bvals * D_intra
    print(L)
    print(bvecs.shape)
    print(n.shape)
    L1 = L * np.dot(bvecs, n) ** 2
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

    am2 = (am / x[2]) ** 2

    summ = np.zeros((len(bvals), len(am)))

    for i in range(len(am)):
        num = (2 * D_intra * am2[i] * small_delta) - 2 + \
              (2 * np.exp(-(D_intra * am2[i] * small_delta))) + \
              (2 * np.exp(-(D_intra * am2[i] * big_delta))) - \
              (np.exp(-(D_intra * am2[i] * (big_delta - small_delta)))) - \
              (np.exp(-(D_intra * am2[i] * (big_delta + small_delta))))

        denom = (D_intra ** 2) * (am2[i] ** 3) * ((x[2]) ** 2 * am2[i] - 1)
        summ[:, i] = num / denom

    summ_rows = np.sum(summ, axis=1)
    g_per = np.zeros(bvals.shape)

    for i in range(len(bvecs)):
        g_per[i] = np.dot(bvecs[i, :], bvecs[i, :]) - \
                   np.dot(bvecs[i, :], n) ** 2

    L2 = 2 * (g_per * gamma ** 2) * summ_rows * G ** 2

    yhat_cylinder = L1 + L2

    # zeppelin
    yhat_zeppelin = bvals * ((D_intra - (D_intra * (1 - x[3]))) *
                             (np.dot(bvecs, n) ** 2) + (D_intra * (1 - x[3])))

    # ball
    yhat_ball = (D_iso * bvals)

    # dot
    yhat_dot = np.dot(bvecs, np.array([0, 0, 0]))

    if debug:
        return L1, summ, summ_rows, g_per, L2, yhat_cylinder, yhat_zeppelin, \
            yhat_ball, yhat_dot
    return yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot


def activax_exvivo_model(x, bvals, bvecs, G, small_delta, big_delta,
                         gamma=gamma,
                         D_intra=0.6 * 10 ** 3, D_iso=2 * 10 ** 3,
                         debug=False):

    """
    Aax_exvivo_nlin

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
    exp(-yhat_cylinder), exp(-yhat_zeppelin), exp(-yhat_ball), exp(-yhat_dot)

    Notes
    --------
    The estimated dMRI normalized signal SActiveAx is assumed to be
    coming from the following four compartments:

    .. math::

        S_hat_ActiveAx = {f1}{S_cylinder(R,theta,phi)}+
                    {f2}{S_zeppelin(d_perp,theta,phi)}+
                    {f3}{S_ball}+
                    {f4}{S_dot}

        where d_perp=D_intra*(1-v)
        S_cylinder = exp(-yhat_cylinder)
        S_zeppelin = exp(-yhat_zeppelin)
        S_ball = exp(-yhat_ball)
        S_dot = exp(-yhat_dot)

    """
    res = activax_exvivo_compartments(x, bvals, bvecs, G, small_delta,
                                      big_delta, gamma=gamma,
                                      D_intra=0.6 * 10 ** 3, D_iso=2 * 10 ** 3,
                                      debug=False)

    yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot = res

    phi = np.vstack([yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot]).T
    phi = np.ascontiguousarray(phi)

    return np.exp(-phi)
