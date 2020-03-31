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
        B[j, :] = (Bxx, Bxy, Bzz, Bxz, Byz, Bzz,
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
