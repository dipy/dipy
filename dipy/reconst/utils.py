import numpy as np


def dki_design_matrix(gtab):
    r"""Construct B design matrix for DKI.

    Parameters
    ----------
    gtab : GradientTable
        Measurement directions.

    Returns
    -------
    B : array (N, 22)
        Design matrix or B matrix for the DKI model
        B[j, :] = (Bxx, Bxy, Byy, Bxz, Byz, Bzz,
                   Bxxxx, Byyyy, Bzzzz, Bxxxy, Bxxxz,
                   Bxyyy, Byyyz, Bxzzz, Byzzz, Bxxyy,
                   Bxxzz, Byyzz, Bxxyz, Bxyyz, Bxyzz,
                   BlogS0)

    """
    b = gtab.bvals
    bvec = gtab.bvecs

    B = np.zeros((len(b), 22))
    B[:, 0] = -b * bvec[:, 0] * bvec[:, 0]
    B[:, 1] = -2 * b * bvec[:, 0] * bvec[:, 1]
    B[:, 2] = -b * bvec[:, 1] * bvec[:, 1]
    B[:, 3] = -2 * b * bvec[:, 0] * bvec[:, 2]
    B[:, 4] = -2 * b * bvec[:, 1] * bvec[:, 2]
    B[:, 5] = -b * bvec[:, 2] * bvec[:, 2]
    B[:, 6] = b * b * bvec[:, 0]**4 / 6
    B[:, 7] = b * b * bvec[:, 1]**4 / 6
    B[:, 8] = b * b * bvec[:, 2]**4 / 6
    B[:, 9] = 4 * b * b * bvec[:, 0]**3 * bvec[:, 1] / 6
    B[:, 10] = 4 * b * b * bvec[:, 0]**3 * bvec[:, 2] / 6
    B[:, 11] = 4 * b * b * bvec[:, 1]**3 * bvec[:, 0] / 6
    B[:, 12] = 4 * b * b * bvec[:, 1]**3 * bvec[:, 2] / 6
    B[:, 13] = 4 * b * b * bvec[:, 2]**3 * bvec[:, 0] / 6
    B[:, 14] = 4 * b * b * bvec[:, 2]**3 * bvec[:, 1] / 6
    B[:, 15] = b * b * bvec[:, 0]**2 * bvec[:, 1]**2
    B[:, 16] = b * b * bvec[:, 0]**2 * bvec[:, 2]**2
    B[:, 17] = b * b * bvec[:, 1]**2 * bvec[:, 2]**2
    B[:, 18] = 2 * b * b * bvec[:, 0]**2 * bvec[:, 1] * bvec[:, 2]
    B[:, 19] = 2 * b * b * bvec[:, 1]**2 * bvec[:, 0] * bvec[:, 2]
    B[:, 20] = 2 * b * b * bvec[:, 2]**2 * bvec[:, 0] * bvec[:, 1]
    B[:, 21] = -np.ones(len(b))

    return B


def cti_design_matrix(gtab1, gtab2):
    r"""Construct B design matrix for CTI.

    Parameters
    ----------
    gtab1: dipy.core.gradients.GradientTable
        A GradientTable class instance for first DDE diffusion epoch
    gtab2: dipy.core.gradients.GradientTable
        A GradientTable class instance for second DDE diffusion epoch

    Returns
    -------
    B: array(N, 43)
    Design matrix or B matrix for the CTI model assuming multiple
    Gaussian Components

    """
    b1 = gtab1.bvals
    b2 = gtab2.bvals
    n1 = gtab1.bvecs
    n2 = gtab2.bvecs
    B = np.zeros((len(b1), 43))

    B[:, 0] = -b1 * (n1[:, 0]**2) - b2 * (n2[:, 0]**2)
    B[:, 1] = -2 * b1 * n1[:, 0] * n1[:, 1] - 2 * b2 * n2[:, 0] * n2[:, 1]
    B[:, 2] = -b1 * n1[:, 1]**2 - b2 * n2[:, 1]**2
    B[:, 3] = -2 * b1 * n1[:, 0] * n1[:, 2] - 2 * b2 * n2[:, 0] * n2[:, 2]
    B[:, 4] = -2 * b1 * n1[:, 1] * n1[:, 2] - 2 * b2 * n2[:, 1] * n2[:, 2]
    B[:, 5] = - b1 * n1[:, 2]**2 - b2 * n2[:, 2]**2
    B[:, 6] = (b1*b1*n1[:, 0]**4 + b2*b2*n2[:, 0]**4)/6
    B[:, 7] = (b1*b1*n1[:, 1]**4 + b2*b2*n2[:, 1]**4)/6
    B[:, 8] = (b1*b1*n1[:, 2]**4 + b2*b2*n2[:, 2]**4)/6
    B[:, 9] = (2*b1 * b1 * n1[:, 0]**3 * n1[:, 1] +
               2 * b2 * b2 * n2[:, 0]**3 * n2[:, 1])/3
    B[:, 10] = (2*b1 * b1 * n1[:, 0]**3 * n1[:, 2] +
                2*b2 * b2 * n2[:, 0]**3 * n2[:, 2])/3
    B[:, 11] = (2*b1 * b1 * n1[:, 1]**3 * n1[:, 0] +
                2*b2 * b2 * n2[:, 1]**3 * n2[:, 0])/3
    B[:, 12] = (2*b1 * b1 * n1[:, 1]**3 * n1[:, 2] +
                2*b2 * b2 * n2[:, 1]**3 * n2[:, 2])/3
    B[:, 13] = (2*b1 * b1 * n1[:, 2]**3 * n1[:, 0] +
                2*b2 * b2 * n2[:, 2]**3 * n2[:, 0])/3
    B[:, 14] = (2*b1 * b1 * n1[:, 2]**3 * n1[:, 1] +
                2*b2 * b2 * n2[:, 2]**3 * n2[:, 1])/3
    B[:, 15] = (b1 * b1 * n1[:, 0]**2 * n1[:, 1]**2 +
                b2*b2*n2[:, 0]**2 * n2[:, 1]**2)
    B[:, 16] = (b1 * b1 * n1[:, 0]**2 * n1[:, 2]**2 +
                b2*b2*n2[:, 0]**2 * n2[:, 2]**2)
    B[:, 17] = (b1 * b1 * n1[:, 1]**2 * n1[:, 2]**2 +
                b2*b2*n2[:, 1]**2 * n2[:, 2]**2)
    B[:, 18] = (2 * b1 * b1 * n1[:, 0] ** 2*n1[:, 1] * n1[:, 2] +
                2*b2*b2*n2[:, 0]**2*n2[:, 1]*n2[:, 2])
    B[:, 19] = (2 * b1 * b1 * n1[:, 1] ** 2*n1[:, 0] * n1[:, 2] +
                2*b2*b2*n2[:, 1]**2*n2[:, 0]*n2[:, 2])
    B[:, 20] = (2 * b1 * b1 * n1[:, 2] ** 2*n1[:, 0] * n1[:, 1] +
                2*b2*b2*n2[:, 2]**2*n2[:, 0]*n2[:, 1])
    B[:, 21] = b1 * n1[:, 0]**2 * b2 * n2[:, 0]**2
    B[:, 22] = b1 * n1[:, 1]**2 * b2 * n2[:, 1]**2
    B[:, 23] = b1 * n1[:, 2]**2 * b2 * n2[:, 2]**2
    B[:, 24] = (b1*n1[:, 1]**2 * b2*n2[:, 2]**2 +
                b1*n1[:, 2]**2 * b2*n2[:, 1]**2)
    B[:, 25] = (b1*n1[:, 0]**2 * b2*n2[:, 2]**2 +
                b1*n1[:, 2]**2 * b2*n2[:, 0]**2)
    B[:, 26] = (b1*n1[:, 0]**2 * b2*n2[:, 1]**2 +
                b1*n1[:, 1]**2 * b2*n2[:, 0]**2)
    B[:, 27] = 2 * (b1*n1[:, 0]**2 * b2*n2[:, 1]*n2[:, 2] +
                    b1*n1[:, 1]*n1[:, 2] * b2*n2[:, 0]**2)
    B[:, 28] = 2 * (b1*n1[:, 0]**2 * b2*n2[:, 0] * n2[:, 2] +
                    b1*n1[:, 0]*n1[:, 2] * b2*n2[:, 0]**2)
    B[:, 29] = 2 * (b1*n1[:, 0]**2 * b2*n2[:, 0] * n2[:, 1] +
                    b1*n1[:, 0]*n1[:, 1] * b2*n2[:, 0]**2)
    B[:, 30] = 2 * (b1*n1[:, 1]**2 * b2*n2[:, 1] * n2[:, 2] +
                    b1*n1[:, 1]*n1[:, 2] * b2*n2[:, 1]**2)
    B[:, 31] = 2 * (b1*n1[:, 1]**2 * b2*n2[:, 0]*n2[:, 2] +
                    b1*n1[:, 0]*n1[:, 2] * b2*n2[:, 1]**2)
    B[:, 32] = 2 * (b1*n1[:, 1]**2 * b2*n2[:, 1] * n2[:, 0] +
                    b1*n1[:, 1]*n1[:, 0] * b2*n2[:, 1]**2)
    B[:, 33] = 2 * (b1*n1[:, 2]**2 * b2*n2[:, 2] * n2[:, 1] +
                    b1*n1[:, 2]*n1[:, 1] * b2*n2[:, 2]**2)
    B[:, 34] = 2 * (b1*n1[:, 2]**2 * b2*n2[:, 2] * n2[:, 0] +
                    b1*n1[:, 2]*n1[:, 0] * b2*n2[:, 2]**2)
    B[:, 35] = 2 * (b1*n1[:, 2]**2 * b2*n2[:, 0]*n2[:, 1] +
                    b1*n1[:, 0]*n1[:, 1] * b2*n2[:, 2]**2)
    B[:, 36] = 4 * (b1*n1[:, 1] * n1[:, 2] *
                    b2*n2[:, 1] * n2[:, 2])
    B[:, 37] = 4 * (b1*n1[:, 0] * n1[:, 2] *
                    b2*n2[:, 0] * n2[:, 2])
    B[:, 38] = 4 * (b1*n1[:, 0] * n1[:, 1] *
                    b2*n2[:, 0] * n2[:, 1])
    B[:, 39] = 4 * (b1*n1[:, 0] * n1[:, 2] * b2*n2[:, 1] * n2[:, 2] +
                    b1*n1[:, 1] * n1[:, 2] * b2*n2[:, 0] * n2[:, 2])
    B[:, 40] = 4 * (b1*n1[:, 0] * n1[:, 1] * b2*n2[:, 0] * n2[:, 2] +
                    b1*n1[:, 0] * n1[:, 2] * b2*n2[:, 0] * n2[:, 1])
    B[:, 41] = 4 * (b1*n1[:, 0] * n1[:, 1] * b2*n2[:, 1] * n2[:, 2] +
                    b1*n1[:, 1] * n1[:, 2] * b2*n2[:, 0] * n2[:, 1])
    B[:, 42] = -np.ones(len(b1))

    return B


def _roi_in_volume(data_shape, roi_center, roi_radii):
    """Ensures that a cuboid ROI is in a data volume.

    Parameters
    ----------
    data_shape : ndarray
        Shape of the data
    roi_center : ndarray, (3,)
        Center of ROI in data.
    roi_radii : ndarray, (3,)
        Radii of cuboid ROI

    Returns
    -------
    roi_radii : ndarray, (3,)
        Truncated radii of cuboid ROI. It remains unchanged if
        the ROI was already contained inside the data volume.
    """

    for i in range(len(roi_center)):
        inf_lim = int(roi_center[i] - roi_radii[i])
        sup_lim = int(roi_center[i] + roi_radii[i])
        if inf_lim < 0 or sup_lim >= data_shape[i]:
            roi_radii[i] = min(int(roi_center[i]),
                               int(data_shape[i] - roi_center[i]))
    return roi_radii


def _mask_from_roi(data_shape, roi_center, roi_radii):
    """Produces a mask from a cuboid ROI defined by center and radii.

    Parameters
    ----------
    data_shape : array-like, (3,)
        Shape of the data from which the ROI is taken.
    roi_center : array-like, (3,)
        Center of ROI in data.
    roi_radii : array-like, (3,)
        Radii of cuboid ROI.

    Returns
    -------
    mask : ndarray
        Mask of the cuboid ROI.
    """

    ci, cj, ck = roi_center
    wi, wj, wk = roi_radii
    interval_i = slice(int(ci - wi), int(ci + wi) + 1)
    interval_j = slice(int(cj - wj), int(cj + wj) + 1)
    interval_k = slice(int(ck - wk), int(ck + wk) + 1)

    if wi == 0:
        interval_i = ci
    elif wj == 0:
        interval_j = cj
    elif wk == 0:
        interval_k = ck

    mask = np.zeros(data_shape, dtype=np.int64)
    mask[interval_i, interval_j, interval_k] = 1

    return mask
