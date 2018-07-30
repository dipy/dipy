""" functions that implement a beltrami framework algorithm to
fit a free water elimination model for single shell datasets """
from __future__ import division
import numpy as np
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table


def F_init(Ak, S0, bval, St, Sw, Diso, lambda_min, lambda_max):
    """ initializes the volume fraction f0 and its limits fmin and fmax """
    # setting fmin and fmax
    A_water = np.exp(-bval * Diso)
    A_min = np.exp(-bval * lambda_max)  # min expected attenuation in tissue
    A_max = np.exp(-bval * lambda_min)  # max expected attenuation in tissue
    A_av = (A_min + A_max) / 2
    min_A = np.min(Ak, axis=-1)  # min observed attenuation
    max_A = np.max(Ak, axis=-1)  # max observed attenuation
    fmin = (min_A - A_water) / (A_max - A_water)
    fmax = (max_A - A_water) / (A_min - A_water)
    fmin[fmin < 0] = 0
    fmin[fmin > 1] = 0.9
    fmax[fmax < 0] = 0.1
    fmax[fmax > 1] = 1

    # setting f0
    f0 = 1 - np.log(np.squeeze(S0) / St) / np.log(Sw / St)
    bad_f0s = np.logical_or(f0 < fmin, f0 > fmax)
    f0[bad_f0s] = (fmax[bad_f0s] + fmin[bad_f0s]) / 2

    return (f0, fmin, fmax)


def D_init(Sk, S0, bval, bvecs, f0, Diso):
    """ initializes the tensor components with regular DTI """
    # Correcting Sk for fwater
    A_water = np.exp(-bval * Diso)
    C_water = (1 - f0) * A_water
    # At = (Ak - C_water[..., np.newaxis]) / f0[..., np.newaxis]
    # At = np.clip(At, 0.0001, 1 - 0.0001)
    St = Sk - S0 * C_water[..., np.newaxis]  # using alternative init

    # creating new gtab and data
    x, y, z, k = St.shape
    bvals = bval * np.ones(k)
    bvals = np.insert(bvals, 0, 0)
    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    gtab = gradient_table(bvals, bvecs)

    init_data = np.zeros((x, y, z, k + 1))
    init_data[..., 0] = S0[..., 0]
    init_data[..., 1:] = St

    # fitting
    model = dti.TensorModel(gtab)
    fit = model.fit(init_data)
    qform = fit.quadratic_form

    Dxx = qform[..., 0, 0]
    Dyy = qform[..., 1, 1]
    Dzz = qform[..., 2, 2]
    Dxy = qform[..., 0, 1]
    Dxz = qform[..., 0, 2]
    Dyz = qform[..., 1, 2]

    return (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)


def tensor_to_iwasawa(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz):
    """
    converts the tensor components of D
    into Iwasawa coordinates field (X, Y, Z, 6)
    """
    X1 = Dxx
    X2 = Dyy - (Dxy**2 / Dxx)
    X3 = ((Dxz**2 * Dyy - 2 * Dxy * Dxz * Dyz + Dxx * Dyz**2) /
          (Dxy**2 - Dxx * Dyy) + Dzz)
    X4 = Dxy / Dxx
    X5 = Dxz / Dxx
    X6 = (Dxy * Dxz - Dxx * Dyz) / (Dxy**2 - Dxx * Dyy)

    return (X1, X2, X3, X4, X5, X6)


def iwasawa_to_tensor(X1, X2, X3, X4, X5, X6):
    """ converts Iwasawa coordinates back to tensor form """
    Dxx = X1
    Dyy = X2 + X1 * X4**2
    Dzz = X3 + X1 * X5**2 + X2 * X6**2
    Dxy = X1 * X4
    Dxz = X1 * X5
    Dyz = X1 * X4 * X5 + X2 * X6

    return (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)


def dfx(X):
    """ forward differnces with respect to x """

    rows = X.shape[0]
    ind = np.insert(np.arange(rows - 1), 0, 0)
    dx = X - X[ind, :, :]

    return dx


def dfy(X):
    """ forward differnces with respect to y """
    cols = X.shape[1]
    ind = np.insert(np.arange(cols - 1), 0, 0)
    dy = X - X[:, ind, :]

    return dy


def dfz(X):
    """ forward differnces with respect to z """
    slices = X.shape[2]
    ind = np.insert(np.arange(slices - 1), 0, 0)
    dz = X - X[:, :, ind]

    return dz


def df(X):
    """ all forward differnces """
    dx = dfx(X)
    dy = dfy(X)
    dz = dfz(X)

    return (dx, dy, dz)


def dbx(X):
    """ backward differnces with respect to x """
    rows = X.shape[0]
    ind = np.append(np.arange(rows)[1:], rows - 1)
    dx = X[ind, :, :] - X

    return dx


def dby(X):
    """ backward differnces with respect to y """
    cols = X.shape[1]
    ind = np.append(np.arange(cols)[1:], cols - 1)
    dy = X[:, ind, :] - X

    return dy


def dbz(X):
    """ backward differnces with respect to z """
    slices = X.shape[2]
    ind = np.append(np.arange(slices)[1:], slices - 1)
    dz = X[:, :, ind] - X

    return dz


def g_metric(X1, X2, X3, X6, dX1, dX2, dX3, dX4, dX5, dX6, beta):
    """ computes the induced metric g of the Iwasawa manifold """
    h44 = 1 / (X1**2)
    h55 = 1 / (X2**2)
    h66 = 1 / (X3**2)
    h77 = (2 * X1 * (X3 + X2 * X6**2)) / (X2 * X3)
    h78 = -(2 * X1 * X6) / X3
    h88 = 2 * X1 / X3
    h99 = 2 * X2 / X3

    dX1x, dX1y, dX1z = dX1
    dX2x, dX2y, dX2z = dX2
    dX3x, dX3y, dX3z = dX3
    dX4x, dX4y, dX4z = dX4
    dX5x, dX5y, dX5z = dX5
    dX6x, dX6y, dX6z = dX6

    g11 = 1 + (dX1x**2 * h44 + dX2x**2 * h55 + dX3x**2 * h66 +
               dX4x**2 * h77 + 2 * dX4x * dX5x * h78 +
               dX5x**2 * h88 + dX6x**2 * h99) * beta

    g22 = 1 + (dX1y**2 * h44 + dX2y**2 * h55 + dX3y**2 * h66 +
               dX4y**2 * h77 + 2 * dX4y * dX5y * h78 +
               dX5y**2 * h88 + dX6y**2 * h99) * beta

    g33 = 1 + (dX1z**2 * h44 + dX2z**2 * h55 + dX3z**2 * h66 +
               dX4z**2 * h77 + 2 * dX4z * dX5z * h78 +
               dX5z**2 * h88 + dX6z**2 * h99) * beta

    g12 = (dX1x * dX1y * h44 + dX2x * dX2y * h55 +
           dX3x * dX3y * h66 + dX4x * dX4y * h77 +
           dX5x * dX4y * h78 + dX4x * dX5y * h78 +
           dX5x * dX5y * h88 + dX6x * dX6y * h99) * beta

    g13 = (dX1x * dX1z * h44 + dX2x * dX2z * h55 +
           dX3x * dX3z * h66 + dX4x * dX4z * h77 +
           dX5x * dX4z * h78 + dX4x * dX5z * h78 +
           dX5x * dX5z * h88 + dX6x * dX6z * h99) * beta

    g23 = (dX1y * dX1z * h44 + dX2y * dX2z * h55 +
           dX3y * dX3z * h66 + dX4y * dX4z * h77 +
           dX5y * dX4z * h78 + dX4y * dX5z * h78 +
           dX5y * dX5z * h88 + dX6y * dX6z * h99) * beta

    return (g11, g22, g33, g12, g13, g23)


def invert_g(X1, X2, X3, X6, dX1, dX2, dX3, dX4, dX5, dX6, beta):
    """ computes the inverse components of metric g """
    g11, g22, g33, g12, g13, g23 = g_metric(X1, X2, X3, X6, dX1, dX2, dX3,
                                            dX4, dX5, dX6, beta)

    inv_g11 = g22 * g33 - g23**2
    inv_g22 = g11 * g33 - g13**2
    inv_g33 = g11 * g22 - g12**2
    inv_g12 = g13 * g23 - g12 * g33
    inv_g13 = g12 * g23 - g13 * g22
    inv_g23 = g12 * g13 - g11 * g23

    gdet = g11 * inv_g11 + g12 * inv_g12 + g13 * inv_g13
    gdet[gdet <= 0] = 0.0001

    g11 = inv_g11 / gdet
    g22 = inv_g22 / gdet
    g33 = inv_g33 / gdet
    g12 = inv_g12 / gdet
    g13 = inv_g13 / gdet
    g23 = inv_g23 / gdet

    g05 = 1 / np.sqrt(gdet)

    return(g11, g22, g33, g12, g13, g23, g05)


def get_A(g11, g22, g33, g12, g13, g23, dXi):
    """
    computes auxiliary A vector field, Ai = ginv * dXi
    used to compute the Laplace-Beltrami and Levi-Civita incrementals
    """
    dXix, dXiy, dXiz = dXi
    x, y, z = g11.shape
    Aix = g11 * dXix + g12 * dXiy + g13 * dXiz
    Aiy = g12 * dXix + g22 * dXiy + g23 * dXiz
    Aiz = g13 * dXix + g23 * dXiy + g33 * dXiz

    return (Aix, Aiy, Aiz)


def laplacian_aux(Ai, g05):
    """
    auxiliary function to compute the Laplace-Beltrami term
    for the ith Iwasawa parameter
    """
    Aix, Aiy, Aiz = Ai
    Lx = dbx(Aix / g05)
    Ly = dby(Aiy / g05)
    Lz = dbz(Aiz / g05)
    Li = g05 * (Lx + Ly + Lz)

    return Li


def laplace_beltrami(A1, A2, A3, A4, A5, A6, g05):
    """ computes the Laplace-Beltrami term for all Iwasawa coordinates """
    L1 = laplacian_aux(A1, g05)
    L2 = laplacian_aux(A2, g05)
    L3 = laplacian_aux(A3, g05)
    L4 = laplacian_aux(A4, g05)
    L5 = laplacian_aux(A5, g05)
    L6 = laplacian_aux(A6, g05)

    return (L1, L2, L3, L4, L5, L6)


def civita_1(X1, X2, X3, X6, dX1, dX4, dX5, A1, A4, A5):
    """ computes the Levi-Civita term for X1 """
    # unpacking
    dX1x, dX1y, dX1z = dX1
    dX4x, dX4y, dX4z = dX4
    dX5x, dX5y, dX5z = dX5

    A1x, A1y, A1z = A1
    A4x, A4y, A4z = A4
    A5x, A5y, A5z = A5

    # computing Christoffel symbols
    Gamma_11 = -1 / X1
    Gamma_44 = - X1**2 * (X3 + X2 * X6**2) / (X2 * X3)
    Gamma_45 = X1**2 * X6 / X3
    Gamma_55 = -X1**2 / X3

    # computing auxiliary B matrix
    B1x = Gamma_11 * dX1x
    B1y = Gamma_11 * dX1y
    B1z = Gamma_11 * dX1z

    B4x = Gamma_44 * dX4x + Gamma_45 * dX5x
    B4y = Gamma_44 * dX4y + Gamma_45 * dX5y
    B4z = Gamma_44 * dX4z + Gamma_45 * dX5z

    B5x = Gamma_45 * dX4x + Gamma_55 * dX5x
    B5y = Gamma_45 * dX4y + Gamma_55 * dX5y
    B5z = Gamma_45 * dX4z + Gamma_55 * dX5z

    # computing the civita term
    C1 = (A1x * B1x + A1y * B1y + A1z * B1z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A5x * B5x + A5y * B5y + A5z * B5z)

    return C1


def civita_2(X1, X2, X3, dX2, dX4, dX6, A2, A4, A6):
    """ computes the Levi-Civita term for X2 """
    # unpacking
    dX2x, dX2y, dX2z = dX2
    dX4x, dX4y, dX4z = dX4
    dX6x, dX6y, dX6z = dX6

    A2x, A2y, A2z = A2
    A4x, A4y, A4z = A4
    A6x, A6y, A6z = A6

    # computing Christoffel symbols
    Gamma_22 = -1 / X2
    Gamma_44 = X1
    Gamma_66 = -X2**2 / X3

    # computing auxiliary B matrix
    B2x = Gamma_22 * dX2x
    B2y = Gamma_22 * dX2y
    B2z = Gamma_22 * dX2z

    B4x = Gamma_44 * dX4x
    B4y = Gamma_44 * dX4y
    B4z = Gamma_44 * dX4z

    B6x = Gamma_66 * dX6x
    B6y = Gamma_66 * dX6y
    B6z = Gamma_66 * dX6z

    # computing the civita term
    C2 = (A2x * B2x + A2y * B2y + A2z * B2z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A6x * B6x + A6y * B6y + A6z * B6z)

    return C2


def civita_3(X1, X2, X3, X6, dX3, dX4, dX5, dX6, A3, A4, A5, A6):
    """ computes the Levi-Civita term for X3 """
    # unpacking
    dX3x, dX3y, dX3z = dX3
    dX4x, dX4y, dX4z = dX4
    dX5x, dX5y, dX5z = dX5
    dX6x, dX6y, dX6z = dX6

    A3x, A3y, A3z = A3
    A4x, A4y, A4z = A4
    A5x, A5y, A5z = A5
    A6x, A6y, A6z = A6

    # computing Christoffel symbols
    Gamma_33 = -1 / X3
    Gamma_44 = X1 * X6**2
    Gamma_45 = -X1 * X6
    Gamma_55 = X1
    Gamma_66 = X2

    # computing auxiliary B matrix
    B3x = Gamma_33 * dX3x
    B3y = Gamma_33 * dX3y
    B3z = Gamma_33 * dX3z

    B4x = Gamma_44 * dX4x + Gamma_45 * dX5x
    B4y = Gamma_44 * dX4y + Gamma_45 * dX5y
    B4z = Gamma_44 * dX4z + Gamma_45 * dX5z

    B5x = Gamma_45 * dX4x + Gamma_55 * dX5x
    B5y = Gamma_45 * dX4y + Gamma_55 * dX5y
    B5z = Gamma_45 * dX4z + Gamma_55 * dX5z

    B6x = Gamma_66 * dX6x
    B6y = Gamma_66 * dX6y
    B6z = Gamma_66 * dX6z

    # computing the civita term
    C3 = (A3x * B3x + A3y * B3y + A3z * B3z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A5x * B5x + A5y * B5y + A5z * B5z +
          A6x * B6x + A6y * B6y + A6z * B6z)

    return C3


def civita_4(X1, X2, X3, X6, dX1, dX2, dX4, dX5, dX6, A1, A2, A4, A5, A6):
    """ computes the Levi-Civita term for X4 """
    # unpacking
    dX1x, dX1y, dX1z = dX1
    dX2x, dX2y, dX2z = dX2
    dX4x, dX4y, dX4z = dX4
    dX5x, dX5y, dX5z = dX5
    dX6x, dX6y, dX6z = dX6

    A1x, A1y, A1z = A1
    A2x, A2y, A2z = A2
    A4x, A4y, A4z = A4
    A5x, A5y, A5z = A5
    A6x, A6y, A6z = A6

    # computing Christoffel symbols
    Gamma_14 = 1 / (2 * X1)
    Gamma_24 = -1 / (2 * X2)
    Gamma_46 = X2 * X6 / (2 * X3)
    Gamma_56 = -X2 / (2 * X3)

    # computing auxiliary B matrix
    B1x = Gamma_14 * dX4x
    B1y = Gamma_14 * dX4y
    B1z = Gamma_14 * dX4z

    B2x = Gamma_24 * dX4x
    B2y = Gamma_24 * dX4y
    B2z = Gamma_24 * dX4z

    B4x = Gamma_14 * dX1x + Gamma_24 * dX2x + Gamma_46 * dX6x
    B4y = Gamma_14 * dX1y + Gamma_24 * dX2y + Gamma_46 * dX6y
    B4z = Gamma_14 * dX1z + Gamma_24 * dX2z + Gamma_46 * dX6z

    B5x = Gamma_56 * dX6x
    B5y = Gamma_56 * dX6y
    B5z = Gamma_56 * dX6z

    B6x = Gamma_46 * dX4x + Gamma_56 * dX5x
    B6y = Gamma_46 * dX4y + Gamma_56 * dX5y
    B6z = Gamma_46 * dX4z + Gamma_56 * dX5z

    # computing the civita term
    C4 = (A1x * B1x + A1y * B1y + A1z * B1z +
          A2x * B2x + A2y * B2y + A2z * B2z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A5x * B5x + A5y * B5y + A5z * B5z +
          A6x * B6x + A6y * B6y + A6z * B6z)

    return C4


def civita_5(X1, X2, X3, X6, dX1, dX2, dX3, dX4, dX5, dX6,
             A1, A2, A3, A4, A5, A6):
    """ computes the Levi-Civita term for X5 """
    # unpacking
    dX1x, dX1y, dX1z = dX1
    dX2x, dX2y, dX2z = dX2
    dX3x, dX3y, dX3z = dX3
    dX4x, dX4y, dX4z = dX4
    dX5x, dX5y, dX5z = dX5
    dX6x, dX6y, dX6z = dX6

    A1x, A1y, A1z = A1
    A2x, A2y, A2z = A2
    A3x, A3y, A3z = A3
    A4x, A4y, A4z = A4
    A5x, A5y, A5z = A5
    A6x, A6y, A6z = A6

    # computing Christoffel symbols
    Gamma_24 = -X6 / (2 * X2)
    Gamma_34 = X6 / (2 * X3)
    Gamma_15 = 1 / (2 * X1)
    Gamma_35 = -1 / (2 * X3)
    Gamma_46 = (-1 + X2 * X6**2 / X3) / 2
    Gamma_56 = -X2 * X6 / (2 * X3)

    # computing auxiliary B matrix
    B1x = Gamma_15 * dX5x
    B1y = Gamma_15 * dX5y
    B1z = Gamma_15 * dX5z

    B2x = Gamma_24 * dX4x
    B2y = Gamma_24 * dX4y
    B2z = Gamma_24 * dX4z

    B3x = Gamma_34 * dX4x + Gamma_35 * dX5x
    B3y = Gamma_34 * dX4y + Gamma_35 * dX5y
    B3z = Gamma_34 * dX4z + Gamma_35 * dX5z

    B4x = Gamma_24 * dX2x + Gamma_34 * dX3x + Gamma_46 * dX6x
    B4y = Gamma_24 * dX2y + Gamma_34 * dX3y + Gamma_46 * dX6y
    B4z = Gamma_24 * dX2z + Gamma_34 * dX3z + Gamma_46 * dX6z

    B5x = Gamma_15 * dX1x + Gamma_35 * dX3x + Gamma_56 * dX6x
    B5y = Gamma_15 * dX1y + Gamma_35 * dX3y + Gamma_56 * dX6y
    B5z = Gamma_15 * dX1z + Gamma_35 * dX3z + Gamma_56 * dX6z

    B6x = Gamma_46 * dX4x + Gamma_56 * dX5x
    B6y = Gamma_46 * dX4y + Gamma_56 * dX5y
    B6z = Gamma_46 * dX4z + Gamma_56 * dX5z

    # computing the civita term
    C5 = (A1x * B1x + A1y * B1y + A1z * B1z +
          A2x * B2x + A2y * B2y + A2z * B2z +
          A3x * B3x + A3y * B3y + A3z * B3z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A5x * B5x + A5y * B5y + A5z * B5z +
          A6x * B6x + A6y * B6y + A6z * B6z)

    return C5


def civita_6(X1, X2, X3, X6, dX2, dX3, dX4, dX5, dX6, A2, A3, A4, A5, A6):
    """ computes the Levi-Civita term for X6 """
    # unpacking
    dX2x, dX2y, dX2z = dX2
    dX3x, dX3y, dX3z = dX3
    dX4x, dX4y, dX4z = dX4
    dX5x, dX5y, dX5z = dX5
    dX6x, dX6y, dX6z = dX6

    A2x, A2y, A2z = A2
    A3x, A3y, A3z = A3
    A4x, A4y, A4z = A4
    A5x, A5y, A5z = A5
    A6x, A6y, A6z = A6

    # computing Christoffel symbols
    Gamma_44 = -X1 * X6 / X2
    Gamma_45 = X1 / (2 * X2)
    Gamma_26 = 1 / (2 * X2)
    Gamma_36 = -1 / (2 * X3)

    # computing auxiliary B matrix
    B2x = Gamma_26 * dX6x
    B2y = Gamma_26 * dX6y
    B2z = Gamma_26 * dX6z

    B3x = Gamma_36 * dX6x + Gamma_45 * dX5x
    B3y = Gamma_36 * dX6y + Gamma_45 * dX5y
    B3z = Gamma_36 * dX6z + Gamma_45 * dX5z

    B4x = Gamma_44 * dX4x
    B4y = Gamma_44 * dX4y
    B4z = Gamma_44 * dX4z

    B5x = Gamma_45 * dX4x
    B5y = Gamma_45 * dX4y
    B5z = Gamma_45 * dX4z

    B6x = Gamma_26 * dX2x + Gamma_36 * dX3x
    B6y = Gamma_26 * dX2y + Gamma_36 * dX3y
    B6z = Gamma_26 * dX2z + Gamma_36 * dX3z

    # computing the civita term
    C6 = (A2x * B2x + A2y * B2y + A2z * B2z +
          A3x * B3x + A3y * B3y + A3z * B3z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A5x * B5x + A5y * B5y + A5z * B5z +
          A6x * B6x + A6y * B6y + A6z * B6z)

    return C6


def levi_civita(X1, X2, X3, X6, dX1, dX2, dX3, dX4, dX5, dX6,
                A1, A2, A3, A4, A5, A6):
    """ computes all Levi-Civita terms """
    C1 = civita_1(X1, X2, X3, X6, dX1, dX4, dX5, A1, A4, A5)
    C2 = civita_2(X1, X2, X3, dX2, dX4, dX6, A2, A4, A6)
    C3 = civita_3(X1, X2, X3, X6, dX3, dX4, dX5, dX6, A3, A4, A5, A6)
    C4 = civita_4(X1, X2, X3, X6, dX1, dX2, dX4, dX5, dX6, A1, A2, A4, A5, A6)
    C5 = civita_5(X1, X2, X3, X6, dX1, dX2, dX3, dX4, dX5, dX6,
                  A1, A2, A3, A4, A5, A6)
    C6 = civita_6(X1, X2, X3, X6, dX2, dX3, dX4, dX5, dX6, A2, A3, A4, A5, A6)

    return (C1, C2, C3, C4, C5, C6)


def get_qDq(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, qk):
    """ get the q*D*q product to compute fidelity term """
    qx = qk[:, 0]
    qy = qk[:, 1]
    qz = qk[:, 2]

    qDq = (Dxx * qx**2 + Dyy * qy**2 + Dzz * qz**2 +
           (Dxy * qx * qy + Dxz * qx * qz + Dyz * qy * qz) * 2)
    qDq = np.clip(qDq, a_min=10**-7, a_max=None)

    return qDq


def fidelity(Ak, X1, X2, X3, X4, X5, X6, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz,
             f, bval, bvecs, g05, alpha, Diso):
    """ computes the fidelity terms for all Iwasawa parameters """
    X1 = X1[..., np.newaxis]
    X2 = X2[..., np.newaxis]
    X3 = X3[..., np.newaxis]
    X4 = X4[..., np.newaxis]
    X5 = X5[..., np.newaxis]
    X6 = X6[..., np.newaxis]

    Dxx = Dxx[..., np.newaxis]
    Dyy = Dyy[..., np.newaxis]
    Dzz = Dzz[..., np.newaxis]
    Dxy = Dxy[..., np.newaxis]
    Dxz = Dxz[..., np.newaxis]
    Dyz = Dyz[..., np.newaxis]

    f = f[..., np.newaxis]

    # computing the part of fidelity common to all X1,..X6
    A_water = np.exp(-bval * Diso)
    C_water = (1 - f) * A_water
    qDq = get_qDq(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, bvecs)
    # qDq = np.clip(qDq, a_min=1 * 10**-7, a_max=None)
    A_tissue = np.exp(-bval * qDq)
    C_tissue = f * A_tissue
    A_bitensor = C_tissue + C_water
    fid_aux = (A_bitensor - Ak) * A_tissue

    # getting all qDXq 's
    qDX1q = get_qDq(1, X4**2, X5**2, X4, X5, X4 * X5, bvecs)
    qDX2q = get_qDq(0, 1, X6**2, 0, 0, X6, bvecs)
    qDX3q = get_qDq(0, 0, 1, 0, 0, 0, bvecs)
    qDX4q = get_qDq(0, 2 * X1 * X4, 0, X1, 0, X1 * X5, bvecs)
    qDX5q = get_qDq(0, 0, 2 * X1 * X5, 0, X1, X1 * X4, bvecs)
    qDX6q = get_qDq(0, 0, 2 * X2 * X6, 0, 0, X2, bvecs)

    # computing total fidelity terms
    F1 = -alpha * bval * g05 * np.sum(fid_aux * qDX1q, axis=-1)
    F2 = -alpha * bval * g05 * np.sum(fid_aux * qDX2q, axis=-1)
    F3 = -alpha * bval * g05 * np.sum(fid_aux * qDX3q, axis=-1)
    F4 = -alpha * bval * g05 * np.sum(fid_aux * qDX4q, axis=-1)
    F5 = -alpha * bval * g05 * np.sum(fid_aux * qDX5q, axis=-1)
    F6 = -alpha * bval * g05 * np.sum(fid_aux * qDX6q, axis=-1)

    return (F1, F2, F3, F4, F5, F6)


def volume_fraction(Ak, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, f, bval, bvecs, Diso):
    """ computes the volume fraction increment """
    Dxx = Dxx[..., np.newaxis]
    Dyy = Dyy[..., np.newaxis]
    Dzz = Dzz[..., np.newaxis]
    Dxy = Dxy[..., np.newaxis]
    Dxz = Dxz[..., np.newaxis]
    Dyz = Dyz[..., np.newaxis]

    f = f[..., np.newaxis]

    A_water = np.exp(-bval * Diso)
    C_water = (1 - f) * A_water
    qDq = get_qDq(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, bvecs)
    A_tissue = np.exp(-bval * qDq)
    C_tissue = f * A_tissue
    A_bitensor = C_tissue + C_water

    delta_f = -bval * np.sum((A_bitensor - Ak) * (A_tissue - A_water), axis=-1)

    return delta_f


def update_X(Ak, X1, X2, X3, X4, X5, X6, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz,
             dX1, dX2, dX3, dX4, dX5, dX6, f, bval, bvecs, alpha, beta, dt,
             Diso):
    """ computes all the incrementals and updates the Iwasawa field """
    # getting g inverse
    g11, g22, g33, g12, g13, g23, g05 = invert_g(X1, X2, X3, X6, dX1, dX2, dX3,
                                                 dX4, dX5, dX6, beta)
    # getting A matrix
    A1 = get_A(g11, g22, g33, g12, g13, g23, dX1)
    A2 = get_A(g11, g22, g33, g12, g13, g23, dX2)
    A3 = get_A(g11, g22, g33, g12, g13, g23, dX3)
    A4 = get_A(g11, g22, g33, g12, g13, g23, dX4)
    A5 = get_A(g11, g22, g33, g12, g13, g23, dX5)
    A6 = get_A(g11, g22, g33, g12, g13, g23, dX6)
    # computing Laplace-Beltrami terms
    L1, L2, L3, L4, L5, L6 = laplace_beltrami(A1, A2, A3, A4, A5, A6, g05)
    # computing Levi-Civita terms
    C1, C2, C3, C4, C5, C6 = levi_civita(X1, X2, X3, X6, dX1, dX2, dX3, dX4,
                                         dX5, dX6, A1, A2, A3, A4, A5, A6)
    # computing fidelity terms
    F1, F2, F3, F4, F5, F6 = fidelity(Ak, X1, X2, X3, X4, X5, X6, Dxx, Dyy,
                                      Dzz, Dxy, Dxz, Dyz, f, bval, bvecs, g05,
                                      alpha, Diso)
    # updating X
    X1n = X1 + dt * (L1 + C1 + F1)
    X2n = X2 + dt * (L2 + C2 + F2)
    X3n = X3 + dt * (L3 + C3 + F3)
    X4n = X4 + dt * (L4 + C4 + F4)
    X5n = X5 + dt * (L5 + C5 + F5)
    X6n = X6 + dt * (L6 + C6 + F6)

    # X1n[X1n < 10**-9] = 10**-9
    # X2n[X2n < 10**-9] = 10**-9
    # X3n[X3n < 10**-9] = 10**-9
    # X4n[X4n < -30] = -30
    # X5n[X5n < -30] = -30
    # X6n[X6n < -30] = -30

    # X1n[X1n > 1] = 1
    # X2n[X2n > 1] = 1
    # X3n[X3n > 1] = 1
    # X4n[X4n > 30] = 30
    # X5n[X5n > 30] = 30
    # X6n[X6n > 30] = 30

    # finding the right limits to the X's is crucial to help the algorithm
    # stay in well behaved zone, the limits were chosen by trial and error,
    # maybe they change with dataset???

    X1n = np.sign(X1n) * np.clip(np.abs(X1n), 10**-4, 0.5)
    X2n = np.sign(X2n) * np.clip(np.abs(X2n), 10**-4, 0.5)
    X3n = np.sign(X3n) * np.clip(np.abs(X3n), 10**-4, 0.5)
    X4n = np.sign(X4n) * np.clip(np.abs(X4n), 10**-4, 0.5)
    X5n = np.sign(X5n) * np.clip(np.abs(X5n), 10**-4, 0.5)
    X6n = np.sign(X6n) * np.clip(np.abs(X6n), 10**-4, 0.5)

    return (X1n, X2n, X3n, X4n, X5n, X6n)


def update_f(Ak, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, f, fmin, fmax, bval, bvecs, dt,
             Diso):
    """ computes the volume fraction incremental and updates """
    df = volume_fraction(Ak, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, f, bval, bvecs,
                         Diso)
    # updating f
    fn = f + dt * df
    # correcting bad values
    bad_fns = np.logical_or(fn < fmin, fn > fmax)
    fn[bad_fns] = f[bad_fns]

    return fn


def beltrami_flow(data, gtab, maxiter, dt, St, Sw, alpha=1, beta=1,
                  Diso=3e-3, lambda_min=0.1e-3, lambda_max=1.3e-3):
    """
    performs gradient descent on single shell DTI data
    to obtain free water map and corrected diffusion components.

    Args:
        data: ndarray (X, Y, Z, K) on which the estimation is to be performed.
        gtab: gradients table class instance.
        maxiter: maximum number of allowed iterations.
        dt: time step / learning rate
        St: intensity of a voxel from S0 image, containing only tissue.
        Sw: intensity of a voxel from S0 image, containing only CSF.
        alpha: fidelity term weight.
        beta: ratio between metric spatial-feature metric h
            and image domain metric g.
        Diso: diffusion constant of free water at body temperature.
        lambda_min: minimum diffusivity expected in tissue.
        lambda_max: maximum diffusivity expected in tissue.

    Returns:
        fw: ndarray (X, Y, Z) of the estimated free water map.
        Dxx, Dyy, Dzz, Dxy, Dxz, Dyz: the corrected diffusion tensor
                                    components.
    """
    # getting S0
    b0s = gtab.b0s_mask
    S0s = data[..., b0s]
    S0 = np.mean(S0s, axis=-1)[..., np.newaxis]
    Sk = data[..., ~b0s]

    # getting Ak
    S0[S0 <= 0] = 0.0001
    Ak = Sk / S0
    Ak = np.clip(Ak, 0.0001, 1 - 0.0001)

    # getting bval and bvecs
    bval = np.unique(gtab.bvals)[1]
    bvecs = gtab.bvecs[~b0s, ...]

    # initializing volume fraction and tensor field
    f0, fmin, fmax = F_init(Ak, S0, bval, St, Sw, Diso, lambda_min, lambda_max)
    D = D_init(Sk, S0, bval, bvecs, f0, Diso)

    fn = np.copy(f0)
    Dn = np.copy(D)
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = Dn

    for i in np.arange(maxiter):
        # if(i == maxiter // 2):
        #     alpha = 0
        # Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = Dn
        X1, X2, X3, X4, X5, X6 = tensor_to_iwasawa(Dxx, Dyy, Dzz, Dxy, Dxz,
                                                   Dyz)
        dX1 = df(X1)
        dX2 = df(X2)
        dX3 = df(X3)
        dX4 = df(X4)
        dX5 = df(X5)
        dX6 = df(X6)

        X1, X2, X3, X4, X5, X6, = update_X(Ak, X1, X2, X3, X4, X5, X6, Dxx,
                                           Dyy, Dzz, Dxy, Dxz, Dyz, dX1, dX2,
                                           dX3, dX4, dX5, dX6, fn, bval, bvecs,
                                           alpha, beta, dt, Diso)
        fn = update_f(Ak, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, fn, fmin, fmax, bval,
                      bvecs, dt, Diso)
        D1, D2, D3, D4, D5, D6 = iwasawa_to_tensor(X1, X2, X3, X4, X5, X6)

        # maybe clipping the D's also helps the stay in well behaved values??

        # D1[D1 < 10**-9] = Dxx[D1 < 10**-9]
        # D2[D2 < 10**-9] = Dyy[D2 < 10**-9]
        # D3[D3 < 10**-9] = Dzz[D3 < 10**-9]
        # D4[D4 < -5*10**-3] = Dxy[D4 < -5*10**-3]
        # D5[D5 < -5*10**-3] = Dxz[D5 < -5*10**-3]
        # D6[D6 < -5*10**-3] = Dyz[D6 < -5*10**-3]

        # D1[D1 > 5*10**-3] = Dxx[D1 > 5*10**-3]
        # D2[D2 > 5*10**-3] = Dyy[D2 > 5*10**-3]
        # D3[D3 > 5*10**-3] = Dzz[D3 > 5*10**-3]
        # D4[D4 > 5*10**-3] = Dxy[D4 > 5*10**-3]
        # D5[D5 > 5*10**-3] = Dxz[D5 > 5*10**-3]
        # D6[D6 > 5*10**-3] = Dyz[D6 > 5*10**-3]

        Dxx = D1
        Dyy = D2
        Dzz = D3
        Dxy = D4
        Dxz = D5
        Dyz = D6

        # Dxx = np.sign(Dxx) * np.clip(np.abs(Dxx), 3*10**-9, 3*10**-2)
        # Dyy = np.sign(Dyy) * np.clip(np.abs(Dyy), 3*10**-9, 3*10**-2)
        # Dzz = np.sign(Dzz) * np.clip(np.abs(Dzz), 3*10**-9, 3*10**-2)
        # Dxy = np.sign(Dxy) * np.clip(np.abs(Dxy), 3*10**-9, 3*10**-3)
        # Dzx = np.sign(Dxz) * np.clip(np.abs(Dxz), 3*10**-9, 3*10**-3)
        # Dyz = np.sign(Dyz) * np.clip(np.abs(Dyz), 3*10**-9, 3*10**-3)

    fw = 1 - fn
    # Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = Dn

    return (fw, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)


def q_form(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz):
    """ converts the output of beltrami flow to quadratic form """

    nx, ny, nz = Dxx.shape
    qform = np.zeros((nx, ny, nz, 3, 3))

    qform[..., 0, 0] = Dxx
    qform[..., 1, 1] = Dyy
    qform[..., 2, 2] = Dzz

    qform[..., 0, 1] = Dxy
    qform[..., 0, 2] = Dxz
    qform[..., 1, 2] = Dyz

    qform[..., 1, 0] = Dxy
    qform[..., 2, 0] = Dxz
    qform[..., 2, 1] = Dyz

    return qform

# # TESTING
# import nibabel as nib
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from dipy.reconst.vec_val_sum import vec_val_vect
# from dipy.io import read_bvals_bvecs
# from dipy.core.gradients import gradient_table
# from dipy.sims import phantom as pht
# from timeit import default_timer as timer

# f_base = '/home/mrk/Documents/code/python/load/simulated_data'
# f_signal = '/simulated_image.nii.gz'
# f_tensor = '/tensor_image.nii.gz'
# f_bvals = '/sim_bvals.txt'
# f_bvecs = '/sim_bvecs.txt'
# f_fw = '/fw_image.nii.gz'

# # multi-shell fwDTI was performed on the cenir_multib dataset provided by dipy
# # to obtain a free water map, using the extraced parameters, a single-shell
# # dataset was simulated with the bitensor model
# sim_img = nib.load(f_base + f_signal)
# sim_data = sim_img.get_data()
# affine = sim_img.affine
# header = sim_img.header
# zooms = header.get_zooms()[:3]
# # --------------------------------------------------------------------------------------
# # RESCALING THE B VALUES, THIS HELPS THE CONVERGENCE!!
# bvals = np.loadtxt(f_base + f_bvals) * 10**-3
# # --------------------------------------------------------------------------------------
# print(bvals)
# bvecs = np.loadtxt(f_base + f_bvecs)
# gtab = gradient_table(bvals, bvecs)

# # sim_data = sim_data[15:45, 20:65, ...]  # truncating data
# sim_data[..., 1:] = pht.add_noise(sim_data[..., 1:], snr=30)  # adding noise
# print(sim_data.shape)

# # fig1 = plt.figure(1)
# # plt.subplot(121)
# # plt.imshow(sim_data[:, :, 2, 10].T, origin='lower', cmap='gray')
# # plt.title('simulated data')
# # plt.subplot(122)
# # plt.imshow(sim_data2[:, :, 2, 10].T, origin='lower', cmap='gray')
# # plt.title('noisy (snr = 30)')
# # plt.show()

# # fig1.savefig('sim_data.png', bbox_inches='tight')
# b0_ind = gtab.b0s_mask
# bval = np.unique(gtab.bvals)[1]
# bvecs = gtab.bvecs[np.logical_not(b0_ind), ...]
# S0 = np.mean(sim_data[:, :, :, b0_ind], axis=-1)
# S0 = S0[..., np.newaxis]
# S0[S0 <= 0] = 0.0001

# plt.figure()
# plt.imshow(S0[..., 2, 0].T, origin='lower', cmap='gray')
# plt.title('S0: choose St and Sw')
# plt.show()

# Sk = sim_data[..., np.logical_not(b0_ind)]
# Ak = Sk / S0
# Ak = np.clip(Ak, 0.0001, 1 - 0.0001)

# Sw = 9 * (10**3)
# St = 2 * (10**3)
# # this value is in units because of the rescaling done to the b-values above
# Diso = 3
# lambda_min = 0.1
# lambda_max = 1.1

# print('computing f0, fmin and fmax...')
# f0, fmin, fmax = F_init(Ak, S0, bval, St, Sw, Diso, lambda_min, lambda_max)
# fw0 = 1 - f0
# fw_min = 1 - fmax
# fw_max = 1 - fmin

# # plotting initial free water
# fig1 = plt.figure(1)
# plt.subplot(131)
# plt.imshow(fw_min[..., 2].T, origin='lower', cmap='gray')
# plt.title('fwater min')

# plt.subplot(132)
# plt.imshow(fw0[..., 2].T, origin='lower', cmap='gray')
# plt.title('initial fwater')

# ax3 = plt.subplot(133)
# im3 = ax3.imshow(fw_max[..., 2].T, origin='lower', cmap='gray')
# ax3.set_title('fwater max')

# plt.subplots_adjust(wspace=0.025, hspace=0.0)
# cax = plt.axes([0.91, 0.17, 0.03, 0.66])
# plt.colorbar(cax=cax)


# print('initializing tensor...')
# D1, D2, D3, D4, D5, D6 = D_init(Sk, S0, bval, bvecs, f0, Diso)
# initial_tensor_field = q_form(D1, D2, D3, D4, D5, D6)

# # print(np.amin(D1))
# # print(np.amin(D2))
# # print(np.amin(D3))
# # print(np.amin(D4))
# # print(np.amin(D5))
# # print(np.amin(D6))
# # print('\n')
# # print(np.amax(D1))
# # print(np.amax(D2))
# # print(np.amax(D3))
# # print(np.amax(D4))
# # print(np.amax(D5))
# # print(np.amax(D6))
# # print('\n')


# print('computing iwasawa field...')
# X1, X2, X3, X4, X5, X6 = tensor_to_iwasawa(D1, D2, D3, D4, D5, D6)

# # print(np.amin(X1))
# # print(np.amin(X2))
# # print(np.amin(X3))
# # print(np.amin(X4))
# # print(np.amin(X5))
# # print(np.amin(X6))
# # print('\n')
# # print(np.amax(X1))
# # print(np.amax(X2))
# # print(np.amax(X3))
# # print(np.amax(X4))
# # print(np.amax(X5))
# # print(np.amax(X6))
# # print('\n')


# print('beltrami flow...')  # TESTING THE ENTIRE ALGORITHM
# data = sim_data
# maxiter = 300
# dt = 0.0006
# Sw = 9 * (10**3)
# St = 2 * (10**3)

# fw, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = beltrami_flow(data, gtab, maxiter, dt, St,
#                                                  Sw, alpha=1, beta=1, Diso=3,
#                                                  lambda_min=0.1, lambda_max=1.1)
# final_tensor_field = q_form(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)

# # initial FA
# evals, evecs = dti.decompose_tensor(initial_tensor_field)
# new_evals, new_evecs = dti.decompose_tensor(final_tensor_field)
# FA0 = dti.fractional_anisotropy(evals)
# FA0[fw > 0.7] = 0

# # final estimated FA
# new_FA = dti.fractional_anisotropy(new_evals)
# new_FA[fw > 0.7] = 0

# # original FA estimated with multi-shell fwdti algorithm
# tensor_img = nib.load(f_base + f_tensor)
# tensor_data = tensor_img.get_data()
# evals, evecs = dti.decompose_tensor(tensor_data)
# original_fa = dti.fractional_anisotropy(evals)
# original_fa[fw > 0.7] = 0

# # plotting  FA maps
# fig2 = plt.figure(2)
# plt.subplot(131)
# plt.imshow(original_fa[..., 2].T, origin='lower', cmap='gray')
# plt.title('original FA')

# plt.subplot(132)
# plt.imshow(FA0[..., 2].T, origin='lower', cmap='gray')
# plt.title('initial FA')

# ax3 = plt.subplot(133)
# im3 = ax3.imshow(new_FA[..., 2].T, origin='lower', cmap='gray')
# ax3.set_title('%i iterations' % maxiter)

# plt.subplots_adjust(wspace=0.025, hspace=0.0)
# cax = plt.axes([0.91, 0.17, 0.03, 0.66])
# plt.colorbar(cax=cax)


# # Free Water
# fw_img = nib.load(f_base + f_fw)
# original_fw = fw_img.get_data()  # estimated with the multi-shell fwdti algorithm

# # plotting final free water
# fig3 = plt.figure(3)
# plt.subplot(131)
# plt.imshow(original_fw[..., 2].T, origin='lower', cmap='gray')
# plt.title('original fwater')

# plt.subplot(132)
# plt.imshow(fw0[..., 2].T, origin='lower', cmap='gray')
# plt.title('initial fwater')

# ax3 = plt.subplot(133)
# im3 = ax3.imshow(fw[..., 2].T, origin='lower', cmap='gray')
# ax3.set_title('%i iterations' % maxiter)

# plt.subplots_adjust(wspace=0.025, hspace=0.0)
# cax = plt.axes([0.91, 0.17, 0.03, 0.66])
# plt.colorbar(cax=cax)

# plt.show()

# # SAVING FIGURES
# # fig1.savefig('FW0.png', bbox_inches='tight')
# # fig2.savefig('final_FA.png', bbox_inches='tight')
# # fig3.savefig('final_FW.png', bbox_inches='tight')

# # print(Dxx.dtype)
# # print(final_tensor_field.dtype)
