from __future__ import division, print_function, absolute_import

from dipy.utils.six import string_types

import numpy as np
try:
    from scipy.linalg import polar
except ImportError:   # Some elderly scipy doesn't have polar
    from dipy.fixes.scipy import polar
from scipy.linalg import inv

from dipy.io import gradients as io
from dipy.core.onetime import auto_attr
from dipy.core.geometry import vector_norm
from dipy.core.sphere import disperse_charges, HemiSphere


class GradientTable(object):
    """Diffusion gradient information

    Parameters
    ----------
    gradients : array_like (N, 3)
        Diffusion gradients. The direction of each of these vectors corresponds
        to the b-vector, and the length corresponds to the b-value.
    b0_threshold : float
        Gradients with b-value less than or equal to `b0_threshold` are
        considered as b0s i.e. without diffusion weighting.

    Attributes
    ----------
    gradients : (N,3) ndarray
        diffusion gradients
    bvals : (N,) ndarray
        The b-value, or magnitude, of each gradient direction.
    qvals: (N,) ndarray
        The q-value for each gradient direction. Needs big and small
        delta.
    bvecs : (N,3) ndarray
        The direction, represented as a unit vector, of each gradient.
    b0s_mask : (N,) ndarray
        Boolean array indicating which gradients have no diffusion
        weighting, ie b-value is close to 0.
    b0_threshold : float
        Gradients with b-value less than or equal to `b0_threshold` are
        considered to not have diffusion weighting.

    See Also
    --------
    gradient_table

    Notes
    --------
    The GradientTable object is immutable. Do NOT assign attributes.
    If you have your gradient table in a bval & bvec format, we recommend
    using the factory function gradient_table

    """
    def __init__(self, gradients, big_delta=None, small_delta=None,
                 b0_threshold=0):
        """Constructor for GradientTable class"""
        gradients = np.asarray(gradients)
        if gradients.ndim != 2 or gradients.shape[1] != 3:
            raise ValueError("gradients should be an (N, 3) array")
        self.gradients = gradients
        # Avoid nan gradients. Set these to 0 instead:
        self.gradients = np.where(np.isnan(gradients), 0., gradients)
        self.big_delta = big_delta
        self.small_delta = small_delta
        self.b0_threshold = b0_threshold

    @auto_attr
    def bvals(self):
        return vector_norm(self.gradients)

    @auto_attr
    def qvals(self):
        tau = self.big_delta - self.small_delta / 3.0
        return np.sqrt(self.bvals / tau) / (2 * np.pi)

    @auto_attr
    def b0s_mask(self):
        return self.bvals <= self.b0_threshold

    @auto_attr
    def bvecs(self):
        # To get unit vectors we divide by bvals, where bvals is 0 we divide by
        # 1 to avoid making nans
        denom = self.bvals + (self.bvals == 0)
        denom = denom.reshape((-1, 1))
        return self.gradients / denom

    @property
    def info(self):
        print('B-values shape (%d,)' % self.bvals.shape)
        print('         min %f ' % self.bvals.min())
        print('         max %f ' % self.bvals.max())
        print('B-vectors shape (%d, %d)' % self.bvecs.shape)
        print('         min %f ' % self.bvecs.min())
        print('         max %f ' % self.bvecs.max())


def gradient_table_from_bvals_bvecs(bvals, bvecs, b0_threshold=0, atol=1e-2,
                                    **kwargs):
    """Creates a GradientTable from a bvals array and a bvecs array

    Parameters
    ----------
    bvals : array_like (N,)
        The b-value, or magnitude, of each gradient direction.
    bvecs : array_like (N, 3)
        The direction, represented as a unit vector, of each gradient.
    b0_threshold : float
        Gradients with b-value less than or equal to `bo_threshold` are
        considered to not have diffusion weighting.
    atol : float
        Each vector in `bvecs` must be a unit vectors up to a tolerance of
        `atol`.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword inputs are passed to GradientTable.

    Returns
    -------
    gradients : GradientTable
        A GradientTable with all the gradient information.

    See Also
    --------
    GradientTable, gradient_table

    """
    bvals = np.asarray(bvals, np.float)
    bvecs = np.asarray(bvecs, np.float)
    dwi_mask = bvals > b0_threshold

    # check that bvals is (N,) array and bvecs is (N, 3) unit vectors
    if bvals.ndim != 1 or bvecs.ndim != 2 or bvecs.shape[0] != bvals.shape[0]:
        raise ValueError("bvals and bvecs should be (N,) and (N, 3) arrays "
                         "respectively, where N is the number of diffusion "
                         "gradients")

    bvecs = np.where(np.isnan(bvecs), 0, bvecs)
    bvecs_close_to_1 = abs(vector_norm(bvecs) - 1) <= atol
    if bvecs.shape[1] != 3:
        raise ValueError("bvecs should be (N, 3)")
    if not np.all(bvecs_close_to_1[dwi_mask]):
        raise ValueError("The vectors in bvecs should be unit (The tolerance "
                         "can be modified as an input parameter)")

    bvecs = np.where(bvecs_close_to_1[:, None], bvecs, 0)
    bvals = bvals * bvecs_close_to_1
    gradients = bvals[:, None] * bvecs

    grad_table = GradientTable(gradients, b0_threshold=b0_threshold, **kwargs)
    grad_table.bvals = bvals
    grad_table.bvecs = bvecs
    grad_table.b0s_mask = ~dwi_mask

    return grad_table


def gradient_table(bvals, bvecs=None, big_delta=None, small_delta=None,
                   b0_threshold=0, atol=1e-2):
    """A general function for creating diffusion MR gradients.

    It reads, loads and prepares scanner parameters like the b-values and
    b-vectors so that they can be useful during the reconstruction process.

    Parameters
    ----------

    bvals : can be any of the four options

        1. an array of shape (N,) or (1, N) or (N, 1) with the b-values.
        2. a path for the file which contains an array like the above (1).
        3. an array of shape (N, 4) or (4, N). Then this parameter is
           considered to be a b-table which contains both bvals and bvecs. In
           this case the next parameter is skipped.
        4. a path for the file which contains an array like the one at (3).

    bvecs : can be any of two options

        1. an array of shape (N, 3) or (3, N) with the b-vectors.
        2. a path for the file which contains an array like the previous.

    big_delta : float
        acquisition timing duration (default None)

    small_delta : float
        acquisition timing duration (default None)

    b0_threshold : float
        All b-values with values less than or equal to `bo_threshold` are
        considered as b0s i.e. without diffusion weighting.

    atol : float
        All b-vectors need to be unit vectors up to a tolerance.

    Returns
    -------
    gradients : GradientTable
        A GradientTable with all the gradient information.

    Examples
    --------
    >>> from dipy.core.gradients import gradient_table
    >>> bvals=1500*np.ones(7)
    >>> bvals[0]=0
    >>> sq2=np.sqrt(2)/2
    >>> bvecs=np.array([[0, 0, 0],
    ...                 [1, 0, 0],
    ...                 [0, 1, 0],
    ...                 [0, 0, 1],
    ...                 [sq2, sq2, 0],
    ...                 [sq2, 0, sq2],
    ...                 [0, sq2, sq2]])
    >>> gt = gradient_table(bvals, bvecs)
    >>> gt.bvecs.shape == bvecs.shape
    True
    >>> gt = gradient_table(bvals, bvecs.T)
    >>> gt.bvecs.shape == bvecs.T.shape
    False

    Notes
    -----
    1. Often b0s (b-values which correspond to images without diffusion
       weighting) have 0 values however in some cases the scanner cannot
       provide b0s of an exact 0 value and it gives a bit higher values
       e.g. 6 or 12. This is the purpose of the b0_threshold in the __init__.
    2. We assume that the minimum number of b-values is 7.
    3. B-vectors should be unit vectors.

    """

    # If you provided strings with full paths, we go and load those from
    # the files:
    if isinstance(bvals, string_types):
        bvals, _ = io.read_bvals_bvecs(bvals, None)
    if isinstance(bvecs, string_types):
        _, bvecs = io.read_bvals_bvecs(None, bvecs)

    bvals = np.asarray(bvals)
    # If bvecs is None we expect bvals to be an (N, 4) or (4, N) array.
    if bvecs is None:
        if bvals.shape[-1] == 4:
            bvecs = bvals[:, 1:]
            bvals = np.squeeze(bvals[:, 0])
        elif bvals.shape[0] == 4:
            bvecs = bvals[1:, :].T
            bvals = np.squeeze(bvals[0, :])
        else:
            raise ValueError("input should be bvals and bvecs OR an (N, 4)"
                             " array containing both bvals and bvecs")
    else:
        bvecs = np.asarray(bvecs)
        if (bvecs.shape[1] > bvecs.shape[0]) and bvecs.shape[0] > 1:
            bvecs = bvecs.T
    return gradient_table_from_bvals_bvecs(bvals, bvecs, big_delta=big_delta,
                                           small_delta=small_delta,
                                           b0_threshold=b0_threshold,
                                           atol=atol)


def reorient_bvecs(gtab, affines):
    """Reorient the directions in a GradientTable.

    When correcting for motion, rotation of the diffusion-weighted volumes
    might cause systematic bias in rotationally invariant measures, such as FA
    and MD, and also cause characteristic biases in tractography, unless the
    gradient directions are appropriately reoriented to compensate for this
    effect [Leemans2009]_.

    Parameters
    ----------
    gtab : GradientTable
        The nominal gradient table with which the data were acquired.
    affines : list or ndarray of shape (n, 4, 4) or (n, 3, 3)
        Each entry in this list or array contain either an affine
        transformation (4,4) or a rotation matrix (3, 3).
        In both cases, the transformations encode the rotation that was applied
        to the image corresponding to one of the non-zero gradient directions
        (ordered according to their order in `gtab.bvecs[~gtab.b0s_mask]`)

    Returns
    -------
    gtab : a GradientTable class instance with the reoriented directions

    References
    ----------
    .. [Leemans2009] The B-Matrix Must Be Rotated When Correcting for
       Subject Motion in DTI Data. Leemans, A. and Jones, D.K. (2009).
       MRM, 61: 1336-1349
    """
    new_bvecs = gtab.bvecs[~gtab.b0s_mask]

    if new_bvecs.shape[0] != len(affines):
        e_s = "Number of affine transformations must match number of "
        e_s += "non-zero gradients"
        raise ValueError(e_s)

    for i, aff in enumerate(affines):
        if aff.shape == (4, 4):
            # This must be an affine!
            # Remove the translation component:
            aff_no_trans = aff[:3, :3]
            # Decompose into rotation and scaling components:
            R, S = polar(aff_no_trans)
        elif aff.shape == (3, 3):
            # We assume this is a rotation matrix:
            R = aff
        Rinv = inv(R)
        # Apply the inverse of the rotation to the corresponding gradient
        # direction:
        new_bvecs[i] = np.dot(Rinv, new_bvecs[i])

    return_bvecs = np.zeros(gtab.bvecs.shape)
    return_bvecs[~gtab.b0s_mask] = new_bvecs
    return gradient_table(gtab.bvals, return_bvecs)


def generate_bvecs(N, iters=5000):
    """Generates N bvectors.

    Uses dipy.core.sphere.disperse_charges to model electrostatic repulsion on
    a unit sphere.

    Parameters
    ----------
    N : int
        The number of bvectors to generate. This should be equal to the number
        of bvals used.
    iters : int
        Number of iterations to run.

    Returns
    -------
    bvecs : (N,3) ndarray
        The generated directions, represented as a unit vector, of each
        gradient.
    """
    theta = np.pi * np.random.rand(N)
    phi = 2 * np.pi * np.random.rand(N)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, iters)
    bvecs = hsph_updated.vertices
    return bvecs


def check_multi_b(gtab, n_bvals, non_zero=True, bmag=None):
    """
    Check if you have enough different b-values in your gradient table

    Parameters
    ----------
    gtab : GradientTable class instance.

    n_bvals : int
        The number of different b-values you are checking for.
    non_zero : bool
        Whether to check only non-zero bvalues. In this case, we will require
        at least `n_bvals` *non-zero* b-values (where non-zero is defined
        depending on the `gtab` object's `b0_threshold` attribute)
    bmag : int
        The order of magnitude of the b-values used. The function will
        normalize the b-values relative $10^{bmag - 1}$. Default: derive this
        value from the maximal b-value provided: $bmag=log_{10}(max(bvals))$.

    Returns
    -------
    bool : Whether there are at least `n_bvals` different b-values in the
    gradient table used.
    """
    bvals = gtab.bvals.copy()
    if non_zero:
        bvals = bvals[~gtab.b0s_mask]

    if bmag is None:
        bmag = int(np.log10(np.max(bvals)))

    b = bvals / (10 ** (bmag - 1))  # normalize b units
    b = b.round()
    uniqueb = np.unique(b)
    if uniqueb.shape[0] < n_bvals:
        return False
    else:
        return True
