import logging
from warnings import warn

import numpy as np
from scipy.linalg import inv, polar

from dipy.io import gradients as io
from dipy.core.onetime import auto_attr
from dipy.core.geometry import vector_norm, vec2vec_rotmat
from dipy.core.sphere import disperse_charges, HemiSphere

from dipy.utils.deprecator import deprecate_with_version


WATER_GYROMAGNETIC_RATIO = 267.513e6  # 1/(sT)

logger = logging.getLogger(__name__)


@deprecate_with_version("dipy.core.gradients.unique_bvals is deprecated, "
                        "Please use "
                        "dipy.core.gradients.unique_bvals_magnitude instead",
                        since='1.2', until='1.4')
def unique_bvals(bvals, bmag=None, rbvals=False):
    """ This function gives the unique rounded b-values of the data

    Parameters
    ----------
    bvals : ndarray
        Array containing the b-values

    bmag : int
        The order of magnitude that the bvalues have to differ to be
        considered an unique b-value. B-values are also rounded up to
        this order of magnitude. Default: derive this value from the
        maximal b-value provided: $bmag=log_{10}(max(bvals)) - 1$.

    rbvals : bool, optional
        If True function also returns all individual rounded b-values.
        Default: False

    Returns
    -------
    ubvals : ndarray
        Array containing the rounded unique b-values
    """
    b = round_bvals(bvals, bmag)
    if rbvals:
        return np.unique(b), b

    return np.unique(b)


def b0_threshold_empty_gradient_message(bvals, idx, b0_threshold):
    """Message about the ``b0_threshold`` value resulting in no gradient
    selection.

    Parameters
    ----------
    bvals : (N,) ndarray
        The b-value, or magnitude, of each gradient direction.
    idx : ndarray
        Indices of the gradients to be selected.
    b0_threshold : float
        Gradients with b-value less than or equal to `b0_threshold` are
        considered to not have diffusion weighting.

    Returns
    -------
    str
        Message.
    """

    return (
        "Filtering gradient values with a b0 threshold value "
        f"of {b0_threshold} results in no gradients being "
        f"selected for the b-values ({bvals[idx]}) corresponding "
        f"to the requested indices ({idx}). Lower the b0 threshold "
        "value.")


def b0_threshold_update_slicing_message(slice_start):
    """Message for b0 threshold value update for slicing.

    Parameters
    ----------
    slice_start : int
        Starting index for slicing.

    Returns
    -------
    str
        Message.
    """

    return f"Updating b0_threshold to {slice_start} for slicing."


def mask_non_weighted_bvals(bvals, b0_threshold):
    """Create a diffusion gradient-weighting mask for the b-values according to
    the provided b0 threshold value.

    Parameters
    ----------
    bvals : (N,) ndarray
        The b-value, or magnitude, of each gradient direction.
    b0_threshold : float
        Gradients with b-value less than or equal to `b0_threshold` are
        considered to not have diffusion weighting.

    Returns
    -------
    ndarray
        Gradient-weighting mask: True for all b-value indices whose value is
        smaller or equal to ``b0_threshold``; False otherwise.
     """

    return bvals <= b0_threshold


class GradientTable:
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
    btens : (N,3,3) ndarray
        The b-tensor of each gradient direction.

    See Also
    --------
    gradient_table

    Notes
    -----
    The GradientTable object is immutable. Do NOT assign attributes.
    If you have your gradient table in a bval & bvec format, we recommend
    using the factory function gradient_table

    """
    def __init__(self, gradients, big_delta=None, small_delta=None,
                 b0_threshold=50, btens=None):
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
        if btens is not None:
            linear_tensor = np.array([[1, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0]])
            planar_tensor = np.array([[0, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]]) / 2
            spherical_tensor = np.array([[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]]) / 3
            cigar_tensor = np.array([[2, 0, 0],
                                     [0, .5, 0],
                                     [0, 0, .5]]) / 3
            if isinstance(btens, str):
                b_tensors = np.zeros((len(self.bvals), 3, 3))
                if btens == 'LTE':
                    b_tensor = linear_tensor
                elif btens == 'PTE':
                    b_tensor = planar_tensor
                elif btens == 'STE':
                    b_tensor = spherical_tensor
                elif btens == 'CTE':
                    b_tensor = cigar_tensor
                else:
                    raise ValueError("%s is an invalid value for btens. "%btens
                                     + "Please provide one of the following: "
                                     + "'LTE', 'PTE', 'STE', 'CTE'.")
                for i, (bvec, bval) in enumerate(zip(self.bvecs, self.bvals)):
                    if btens == 'STE':
                        b_tensors[i] = b_tensor * bval
                    else:
                        R = vec2vec_rotmat(np.array([1, 0, 0]), bvec)
                        b_tensors[i] = (np.matmul(np.matmul(R, b_tensor), R.T)
                                        * bval)
                self.btens = b_tensors
            elif (isinstance(btens, np.ndarray)
                  and btens.shape in ((gradients.shape[0],),
                                      (gradients.shape[0], 1),
                                      (1, gradients.shape[0]))):
                b_tensors = np.zeros((len(self.bvals), 3, 3))
                if btens.shape == (1, gradients.shape[0]):
                    btens = btens.reshape((gradients.shape[0], 1))
                for i, (bvec, bval) in enumerate(zip(self.bvecs, self.bvals)):
                    R = vec2vec_rotmat(np.array([1, 0, 0]), bvec)
                    if btens[i] == 'LTE':
                        b_tensors[i] = (np.matmul(np.matmul(R, linear_tensor),
                                        R.T) * bval)
                    elif btens[i] == 'PTE':
                        b_tensors[i] = (np.matmul(np.matmul(R, planar_tensor),
                                        R.T) * bval)
                    elif btens[i] == 'STE':
                        b_tensors[i] = spherical_tensor * bval
                    elif btens[i] == 'CTE':
                        b_tensors[i] = (np.matmul(np.matmul(R, cigar_tensor),
                                        R.T) * bval)
                    else:
                        raise ValueError(
                                "%s is an invalid value in btens. "%btens[i]
                                + "Array element options: 'LTE', 'PTE', 'STE', "
                                + "'CTE'.")
                self.btens = b_tensors
            elif (isinstance(btens, np.ndarray) and btens.shape ==
                    (gradients.shape[0], 3, 3)):
                self.btens = btens
            else:
                raise ValueError("%s is an invalid value for btens. "%btens
                                 + "Please provide a string, an array of "
                                 + "strings, or an array of exact b-tensors. "
                                 + "String options: 'LTE', 'PTE', 'STE', 'CTE'")
        else:
            self.btens = None

    @auto_attr
    def bvals(self):
        return vector_norm(self.gradients)

    @auto_attr
    def tau(self):
        return self.big_delta - self.small_delta / 3.0

    @auto_attr
    def qvals(self):
        tau = self.big_delta - self.small_delta / 3.0
        return np.sqrt(self.bvals / tau) / (2 * np.pi)

    @auto_attr
    def gradient_strength(self):
        tau = self.big_delta - self.small_delta / 3.0
        qvals = np.sqrt(self.bvals / tau) / (2 * np.pi)
        gradient_strength = (qvals * (2 * np.pi) /
                             (self.small_delta * WATER_GYROMAGNETIC_RATIO))
        return gradient_strength

    @auto_attr
    def b0s_mask(self):
        return mask_non_weighted_bvals(self.bvals, self.b0_threshold)

    @auto_attr
    def bvecs(self):
        # To get unit vectors we divide by bvals, where bvals is 0 we divide by
        # 1 to avoid making nans
        denom = self.bvals + (self.bvals == 0)
        denom = denom.reshape((-1, 1))
        return self.gradients / denom

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]  # convert in a list if integer.

        elif isinstance(idx, slice):
            # Get the lower bound of the slice
            slice_start = idx.start if idx.start is not None else 0
            # Check if it is different from b0_threshold
            if slice_start != self.b0_threshold:
                # Update b0_threshold and warn the user
                self.b0_threshold = slice_start
                warn(b0_threshold_update_slicing_message(slice_start),
                     UserWarning, stacklevel=2)
                idx = range(*idx.indices(len(self.bvals)))

        mask = np.logical_not(
            mask_non_weighted_bvals(self.bvals[idx], self.b0_threshold))
        if not any(mask):
            raise ValueError(
                b0_threshold_empty_gradient_message(
                    self.bvals, idx, self.b0_threshold)
            )

        # Apply the mask to select the desired b-values and b-vectors
        bvals_selected = self.bvals[idx][mask]
        bvecs_selected = self.bvecs[idx, :][mask, :]

        # Create a new MyGradientTable object with the selected b-values
        # and b-vectors
        return gradient_table_from_bvals_bvecs(bvals_selected,
                                               bvecs_selected,
                                               big_delta=self.big_delta,
                                               small_delta=self.small_delta,
                                               b0_threshold=self.b0_threshold,
                                               btens=self.btens)
        # removing atol parameter as it's not in the constructor
        # of GradientTable.

    @property
    def info(self, use_logging=False):
        show = logging.info if use_logging else print
        show(self.__str__())

    def __str__(self):
        msg = 'B-values shape {}\n'.format(self.bvals.shape)
        msg += '         min {:f}\n'.format(self.bvals.min())
        msg += '         max {:f}\n'.format(self.bvals.max())
        msg += 'B-vectors shape {}\n'.format(self.bvecs.shape)
        msg += '          min {:f}\n'.format(self.bvecs.min())
        msg += '          max {:f}\n'.format(self.bvecs.max())
        return msg


def gradient_table_from_bvals_bvecs(bvals, bvecs, b0_threshold=50, atol=1e-2,
                                    btens=None, **kwargs):
    """Creates a GradientTable from a bvals array and a bvecs array

    Parameters
    ----------
    bvals : array_like (N,)
        The b-value, or magnitude, of each gradient direction.
    bvecs : array_like (N, 3)
        The direction, represented as a unit vector, of each gradient.
    b0_threshold : float
        Gradients with b-value less than or equal to `b0_threshold` are
        considered to not have diffusion weighting. If its value is equal to or
        larger than all values in b-vals, then it is assumed that no
        thresholding is requested.
    atol : float
        Each vector in `bvecs` must be a unit vectors up to a tolerance of
        `atol`.
    btens : can be any of three options
        1. a string specifying the shape of the encoding tensor for all volumes
           in data. Options: 'LTE', 'PTE', 'STE', 'CTE' corresponding to
           linear, planar, spherical, and "cigar-shaped" tensor encoding.
           Tensors are rotated so that linear and cigar tensors are aligned
           with the corresponding gradient direction and the planar tensor's
           normal is aligned with the corresponding gradient direction.
           Magnitude is scaled to match the b-value.
        2. an array of strings of shape (N,), (N, 1), or (1, N) specifying
           encoding tensor shape for each volume separately. N corresponds to
           the number volumes in data. Options for elements in array: 'LTE',
           'PTE', 'STE', 'CTE' corresponding to linear, planar, spherical, and
           "cigar-shaped" tensor encoding. Tensors are rotated so that linear
           and cigar tensors are aligned with the corresponding gradient
           direction and the planar tensor's normal is aligned with the
           corresponding gradient direction. Magnitude is scaled to match the
           b-value.
        3. an array of shape (N,3,3) specifying the b-tensor of each volume
           exactly. N corresponds to the number volumes in data. No rotation or
           scaling is performed.

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
    bvals = np.asarray(bvals, float)
    bvecs = np.asarray(bvecs, float)
    dwi_mask = bvals > b0_threshold

    # check that bvals is (N,) array and bvecs is (N, 3) unit vectors
    if bvals.ndim != 1 or bvecs.ndim != 2 or bvecs.shape[0] != bvals.shape[0]:
        raise ValueError("bvals and bvecs should be (N,) and (N, 3) arrays "
                         "respectively, where N is the number of diffusion "
                         "gradients")
    # checking for negative bvals
    if b0_threshold < 0:
        raise ValueError("Negative bvals in the data are not feasible")

    # Upper bound for the b0_threshold
    if b0_threshold >= 200:
        warn("b0_threshold has a value > 199")

    # If all b-values are smaller or equal to the b0 threshold, it is assumed
    # that no thresholding is requested
    if any(mask_non_weighted_bvals(bvals, b0_threshold)):
        # checking for the correctness of bvals
        if b0_threshold < bvals.min():
            warn("b0_threshold (value: {0}) is too low, increase your "
                 "b0_threshold. It should be higher than the lowest b0 value "
                 "({1}).".format(b0_threshold, bvals.min()))

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

    grad_table = GradientTable(gradients, b0_threshold=b0_threshold,
                               btens=btens, **kwargs)
    grad_table.bvals = bvals
    grad_table.bvecs = bvecs
    grad_table.b0s_mask = ~dwi_mask

    return grad_table


def gradient_table_from_qvals_bvecs(qvals, bvecs, big_delta, small_delta,
                                    b0_threshold=50, atol=1e-2):
    """A general function for creating diffusion MR gradients.

    It reads, loads and prepares scanner parameters like the b-values and
    b-vectors so that they can be useful during the reconstruction process.

    Parameters
    ----------

    qvals : an array of shape (N,),
        q-value given in 1/mm

    bvecs : can be any of two options

        1. an array of shape (N, 3) or (3, N) with the b-vectors.
        2. a path for the file which contains an array like the previous.

    big_delta : float or array of shape (N,)
        acquisition pulse separation time in seconds

    small_delta : float
        acquisition pulse duration time in seconds

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
    >>> from dipy.core.gradients import gradient_table_from_qvals_bvecs
    >>> qvals = 30. * np.ones(7)
    >>> big_delta = .03  # pulse separation of 30ms
    >>> small_delta = 0.01  # pulse duration of 10ms
    >>> qvals[0] = 0
    >>> sq2 = np.sqrt(2) / 2
    >>> bvecs = np.array([[0, 0, 0],
    ...                   [1, 0, 0],
    ...                   [0, 1, 0],
    ...                   [0, 0, 1],
    ...                   [sq2, sq2, 0],
    ...                   [sq2, 0, sq2],
    ...                   [0, sq2, sq2]])
    >>> gt = gradient_table_from_qvals_bvecs(qvals, bvecs,
    ...                                      big_delta, small_delta)

    Notes
    -----
    1. Often b0s (b-values which correspond to images without diffusion
       weighting) have 0 values however in some cases the scanner cannot
       provide b0s of an exact 0 value and it gives a bit higher values
       e.g. 6 or 12. This is the purpose of the b0_threshold in the __init__.
    2. We assume that the minimum number of b-values is 7.
    3. B-vectors should be unit vectors.

    """
    qvals = np.asarray(qvals)
    bvecs = np.asarray(bvecs)

    if (bvecs.shape[1] > bvecs.shape[0]) and bvecs.shape[0] > 1:
        bvecs = bvecs.T
    bvals = (qvals * 2 * np.pi) ** 2 * (big_delta - small_delta / 3.)
    return gradient_table_from_bvals_bvecs(bvals, bvecs, big_delta=big_delta,
                                           small_delta=small_delta,
                                           b0_threshold=b0_threshold,
                                           atol=atol)


def gradient_table_from_gradient_strength_bvecs(gradient_strength, bvecs,
                                                big_delta, small_delta,
                                                b0_threshold=50, atol=1e-2):
    """A general function for creating diffusion MR gradients.

    It reads, loads and prepares scanner parameters like the b-values and
    b-vectors so that they can be useful during the reconstruction process.

    Parameters
    ----------

    gradient_strength : an array of shape (N,),
        gradient strength given in T/mm

    bvecs : can be any of two options

        1. an array of shape (N, 3) or (3, N) with the b-vectors.
        2. a path for the file which contains an array like the previous.

    big_delta : float or array of shape (N,)
        acquisition pulse separation time in seconds

    small_delta : float
        acquisition pulse duration time in seconds

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
    >>> from dipy.core.gradients import (
    ...    gradient_table_from_gradient_strength_bvecs)
    >>> gradient_strength = .03e-3 * np.ones(7)  # clinical strength at 30 mT/m
    >>> big_delta = .03  # pulse separation of 30ms
    >>> small_delta = 0.01  # pulse duration of 10ms
    >>> gradient_strength[0] = 0
    >>> sq2 = np.sqrt(2) / 2
    >>> bvecs = np.array([[0, 0, 0],
    ...                   [1, 0, 0],
    ...                   [0, 1, 0],
    ...                   [0, 0, 1],
    ...                   [sq2, sq2, 0],
    ...                   [sq2, 0, sq2],
    ...                   [0, sq2, sq2]])
    >>> gt = gradient_table_from_gradient_strength_bvecs(
    ...     gradient_strength, bvecs, big_delta, small_delta)

    Notes
    -----
    1. Often b0s (b-values which correspond to images without diffusion
       weighting) have 0 values however in some cases the scanner cannot
       provide b0s of an exact 0 value and it gives a bit higher values
       e.g. 6 or 12. This is the purpose of the b0_threshold in the __init__.
    2. We assume that the minimum number of b-values is 7.
    3. B-vectors should be unit vectors.

    """
    gradient_strength = np.asarray(gradient_strength)
    bvecs = np.asarray(bvecs)
    if (bvecs.shape[1] > bvecs.shape[0]) and bvecs.shape[0] > 1:
        bvecs = bvecs.T
    qvals = gradient_strength * small_delta * WATER_GYROMAGNETIC_RATIO /\
        (2 * np.pi)
    bvals = (qvals * 2 * np.pi) ** 2 * (big_delta - small_delta / 3.)
    return gradient_table_from_bvals_bvecs(bvals, bvecs, big_delta=big_delta,
                                           small_delta=small_delta,
                                           b0_threshold=b0_threshold,
                                           atol=atol)


def gradient_table(bvals, bvecs=None, big_delta=None, small_delta=None,
                   b0_threshold=50, atol=1e-2, btens=None):
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
        acquisition pulse separation time in seconds (default None)

    small_delta : float
        acquisition pulse duration time in seconds (default None)

    b0_threshold : float
        All b-values with values less than or equal to `bo_threshold` are
        considered as b0s i.e. without diffusion weighting.

    atol : float
        All b-vectors need to be unit vectors up to a tolerance.

    btens : can be any of three options

        1. a string specifying the shape of the encoding tensor for all volumes
           in data. Options: 'LTE', 'PTE', 'STE', 'CTE' corresponding to
           linear, planar, spherical, and "cigar-shaped" tensor encoding.
           Tensors are rotated so that linear and cigar tensors are aligned
           with the corresponding gradient direction and the planar tensor's
           normal is aligned with the corresponding gradient direction.
           Magnitude is scaled to match the b-value.
        2. an array of strings of shape (N,), (N, 1), or (1, N) specifying
           encoding tensor shape for each volume separately. N corresponds to
           the number volumes in data. Options for elements in array: 'LTE',
           'PTE', 'STE', 'CTE' corresponding to linear, planar, spherical, and
           "cigar-shaped" tensor encoding. Tensors are rotated so that linear
           and cigar tensors are aligned with the corresponding gradient
           direction and the planar tensor's normal is aligned with the
           corresponding gradient direction. Magnitude is scaled to match the
           b-value.
        3. an array of shape (N,3,3) specifying the b-tensor of each volume
           exactly. N corresponds to the number volumes in data. No rotation or
           scaling is performed.

    Returns
    -------
    gradients : GradientTable
        A GradientTable with all the gradient information.

    Examples
    --------
    >>> from dipy.core.gradients import gradient_table
    >>> bvals = 1500 * np.ones(7)
    >>> bvals[0] = 0
    >>> sq2 = np.sqrt(2) / 2
    >>> bvecs = np.array([[0, 0, 0],
    ...                   [1, 0, 0],
    ...                   [0, 1, 0],
    ...                   [0, 0, 1],
    ...                   [sq2, sq2, 0],
    ...                   [sq2, 0, sq2],
    ...                   [0, sq2, sq2]])
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
    if isinstance(bvals, str):
        bvals, _ = io.read_bvals_bvecs(bvals, None)
    if isinstance(bvecs, str):
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
        if bvecs.shape[1] != 3 and bvecs.shape[0] > 1:
            bvecs = bvecs.T
    return gradient_table_from_bvals_bvecs(bvals, bvecs, big_delta=big_delta,
                                           small_delta=small_delta,
                                           b0_threshold=b0_threshold,
                                           atol=atol, btens=btens)


def reorient_bvecs(gtab, affines, atol=1e-2):
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
    affines : list or ndarray of shape (4, 4, n) or (3, 3, n)
        Each entry in this list or array contain either an affine
        transformation (4,4) or a rotation matrix (3, 3).
        In both cases, the transformations encode the rotation that was applied
        to the image corresponding to one of the non-zero gradient directions
        (ordered according to their order in `gtab.bvecs[~gtab.b0s_mask]`)
    atol: see gradient_table()

    Returns
    -------
    gtab : a GradientTable class instance with the reoriented directions

    References
    ----------
    .. [Leemans2009] The B-Matrix Must Be Rotated When Correcting for
       Subject Motion in DTI Data. Leemans, A. and Jones, D.K. (2009).
       MRM, 61: 1336-1349
    """
    if isinstance(affines, list):
        affines = np.stack(affines, axis=-1)

    if affines.shape[0] != affines.shape[1]:
        msg = '''reorient_bvecs has changed to require affines as
        (4, 4, n) or (3, 3, n). Shape of (n, 4, 4) or (n, 3, 3)
        is deprecated and will fail in the future.'''
        warn(msg, UserWarning)
        affines = np.moveaxis(affines, 0, -1)

    new_bvecs = gtab.bvecs[~gtab.b0s_mask]

    if new_bvecs.shape[0] != affines.shape[-1]:
        e_s = "Number of affine transformations must match number of "
        e_s += "non-zero gradients"
        raise ValueError(e_s)

    # moving axis to make life easier
    affines = np.moveaxis(affines, -1, 0)

    for i, aff in enumerate(affines):
        if aff.shape == (4, 4):
            # This must be an affine!
            # Remove the translation component:
            aff = aff[:3, :3]
        # Decompose into rotation and scaling components:
        R, S = polar(aff)
        Rinv = inv(R)
        # Apply the inverse of the rotation to the corresponding gradient
        # direction:
        new_bvecs[i] = np.dot(Rinv, new_bvecs[i])

    return_bvecs = np.zeros(gtab.bvecs.shape)
    return_bvecs[~gtab.b0s_mask] = new_bvecs
    return gradient_table(gtab.bvals, return_bvecs, big_delta=gtab.big_delta,
                          small_delta=gtab.small_delta,
                          b0_threshold=gtab.b0_threshold, atol=atol)


def generate_bvecs(N, iters=5000, rng=None):
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
    rng : numpy.random.Generator, optional
        Numpy's random number generator. If None, the generator is created.
        Default is None.

    Returns
    -------
    bvecs : (N,3) ndarray
        The generated directions, represented as a unit vector, of each
        gradient.
    """
    if rng is None:
        rng = np.random.default_rng()
    theta = np.pi * rng.random(N)
    phi = 2 * np.pi * rng.random(N)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, iters)
    bvecs = hsph_updated.vertices
    return bvecs


def round_bvals(bvals, bmag=None):
    """"This function rounds the b-values

    Parameters
    ----------
    bvals : ndarray
        Array containing the b-values

    bmag : int
        The order of magnitude to round the b-values. If not given b-values
        will be rounded relative to the order of magnitude
        $bmag = (bmagmax - 1)$, where bmaxmag is the magnitude order of the
        larger b-value.

    Returns
    -------
    rbvals : ndarray
        Array containing the rounded b-values
    """
    if bmag is None:
        bmag = int(np.log10(np.max(bvals))) - 1

    b = bvals / (10 ** bmag)  # normalize b units
    return b.round() * (10 ** bmag)


def unique_bvals_tolerance(bvals, tol=20):
    """ Gives the unique b-values of the data, within a tolerance gap

    The b-values must be regrouped in clusters easily separated by a
    distance greater than the tolerance gap. If all the b-values of a
    cluster fit within the tolerance gap, the highest b-value is kept.

    Parameters
    ----------
    bvals : ndarray
        Array containing the b-values

    tol : int
        The tolerated gap between the b-values to extract
        and the actual b-values.

    Returns
    -------
    ubvals : ndarray
        Array containing the unique b-values using the median value
        for each cluster
    """
    b = np.unique(bvals)
    ubvals = []
    lower_part = np.where(b <= b[0] + tol)[0]
    upper_part = np.where(np.logical_and(b <= b[lower_part[-1]] + tol,
                                         b > b[lower_part[-1]]))[0]
    ubvals.append(b[lower_part[-1]])
    if len(upper_part) != 0:
        b_index = upper_part[-1] + 1
    else:
        b_index = lower_part[-1] + 1
    while b_index != len(b):
        lower_part = np.where(np.logical_and(b <= b[b_index] + tol,
                                             b > b[b_index-1]))[0]
        upper_part = np.where(np.logical_and(b <= b[lower_part[-1]] + tol,
                                             b > b[lower_part[-1]]))[0]
        ubvals.append(b[lower_part[-1]])
        if len(upper_part) != 0:
            b_index = upper_part[-1] + 1
        else:
            b_index = lower_part[-1] + 1

    # Checking for overlap with get_bval_indices
    for i, ubval in enumerate(ubvals[:-1]):
        indices_1 = get_bval_indices(bvals, ubval, tol)
        indices_2 = get_bval_indices(bvals, ubvals[i+1], tol)
        if len(np.intersect1d(indices_1, indices_2)) != 0:
            msg = '''There is overlap in clustering of b-values.
            The tolerance factor might be too high.'''
            warn(msg, UserWarning)

    return np.asarray(ubvals)


def get_bval_indices(bvals, bval, tol=20):
    """
    Get indices where the b-value is `bval`

    Parameters
    ----------
    bvals: ndarray
        Array containing the b-values

    bval: float or int
        b-value to extract indices

    tol: int
        The tolerated gap between the b-values to extract
        and the actual b-values.

    Returns
    -------
    Array of indices where the b-value is `bval`
    """
    return np.where(np.logical_and(bvals <= bval + tol,
                                   bvals >= bval - tol))[0]


def unique_bvals_magnitude(bvals, bmag=None, rbvals=False):
    """ This function gives the unique rounded b-values of the data

    Parameters
    ----------
    bvals : ndarray
        Array containing the b-values

    bmag : int
        The order of magnitude that the bvalues have to differ to be
        considered an unique b-value. B-values are also rounded up to
        this order of magnitude. Default: derive this value from the
        maximal b-value provided: $bmag=log_{10}(max(bvals)) - 1$.

    rbvals : bool, optional
        If True function also returns all individual rounded b-values.
        Default: False

    Returns
    -------
    ubvals : ndarray
        Array containing the rounded unique b-values
    """
    b = round_bvals(bvals, bmag)
    if rbvals:
        return np.unique(b), b

    return np.unique(b)


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
        normalize the b-values relative $10^{bmag}$. Default: derive this
        value from the maximal b-value provided:
        $bmag=log_{10}(max(bvals)) - 1$.

    Returns
    -------
    bool : Whether there are at least `n_bvals` different b-values in the
    gradient table used.
    """
    bvals = gtab.bvals.copy()
    if non_zero:
        bvals = bvals[~gtab.b0s_mask]

    uniqueb = unique_bvals_magnitude(bvals, bmag=bmag)
    if uniqueb.shape[0] < n_bvals:
        return False
    else:
        return True


def _btens_to_params_2d(btens_2d, ztol):
    """Compute trace, anisotropy and asymmetry parameters from a single b-tensor

    Auxiliary function where calculation of `bval`, bdelta` and `b_eta` from a
    (3,3) b-tensor takes place. The main function `btens_to_params` then wraps
    around this to enable support of input (N, 3, 3) arrays, where N = number of
    b-tensors

    Parameters
    ----------
    btens_2d : (3, 3) numpy.ndarray
        input b-tensor
    ztol : float
        Any parameters smaller than this value are considered to be 0

    Returns
    -------
    bval: float
        b-value (trace)
    bdelta: float
        normalized tensor anisotropy
    bdelta: float
        tensor asymmetry

    Notes
    -----
    Implementation following [1].

    References
    ----------
    .. [1] D. Topgaard, NMR methods for studying microscopic diffusion
    anisotropy, in: R. Valiullin (Ed.), Diffusion NMR of Confined Systems: Fluid
    Transport in Porous Solids and Heterogeneous Materials, Royal Society of
    Chemistry, Cambridge, UK, 2016.

    """
    btens_2d[abs(btens_2d) <= ztol] = 0

    evals = np.real(np.linalg.eig(btens_2d)[0])
    bval = np.sum(evals)
    bval_is_zero = np.abs(bval) <= ztol

    if bval_is_zero:
        bval = 0
        bdelta = 0
        b_eta = 0
    else:
        lambda_iso = (1/3)*bval

        diff_lambdas = np.abs(evals - lambda_iso)
        evals_zzxxyy = evals[np.argsort(diff_lambdas)[::-1]]

        lambda_zz = evals_zzxxyy[0]
        lambda_xx = evals_zzxxyy[1]
        lambda_yy = evals_zzxxyy[2]

        bdelta = (1/(3*lambda_iso))*(lambda_zz-((lambda_yy+lambda_xx)/2))

        if np.abs(bdelta) <= ztol:
            bdelta = 0

        yyxx_diff = lambda_yy-lambda_xx
        if abs(yyxx_diff) <= np.spacing(1):
            yyxx_diff = 0

        b_eta = yyxx_diff/(2*lambda_iso*bdelta+np.spacing(1))

        if np.abs(b_eta) <= b_eta:
            b_eta = 0

    return float(bval), float(bdelta), float(b_eta)


def btens_to_params(btens, ztol=1e-10):
    r"""Compute trace, anisotropy and asymmetry parameters from b-tensors.

    Parameters
    ----------
    btens : (3, 3) OR (N, 3, 3) numpy.ndarray
        input b-tensor, or b-tensors, where N = number of b-tensors
    ztol : float
        Any parameters smaller than this value are considered to be 0

    Returns
    -------
    bval: numpy.ndarray
        b-value(s) (trace(s))
    bdelta: numpy.ndarray
        normalized tensor anisotropy(s)
    b_eta: numpy.ndarray
        tensor asymmetry(s)

    Notes
    -----
    This function can be used to get b-tensor parameters directly from the
    GradientTable `btens` attribute.

    Examples
    --------
    >>> lte = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    >>> bval, bdelta, b_eta = btens_to_params(lte)
    >>> print("bval={}; bdelta={}; b_eta={}".format(bdelta, bval, b_eta))
    bval=[ 1.]; bdelta=[ 1.]; b_eta=[ 0.]

    """
    # Bad input checks
    value_error_msg = "`btens` must be a 2D or 3D numpy array, respectively" \
                      " with (3, 3) or (N, 3, 3) shape, where N corresponds" \
                      " to the number of b-tensors"
    if not isinstance(btens, np.ndarray):
        raise ValueError(value_error_msg)

    nd = np.ndim(btens)
    if nd == 2:
        btens_shape = btens.shape
    elif nd == 3:
        btens_shape = btens.shape[1:]
    else:
        raise ValueError(value_error_msg)

    if not btens_shape == (3, 3):
        raise ValueError(value_error_msg)

    # Reshape so that loop below works when only one input b-tensor is provided
    if nd == 2:
        btens = btens.reshape((1, 3, 3))

    # Pre-allocate
    n_btens = btens.shape[0]
    bval = np.empty(n_btens)
    bdelta = np.empty(n_btens)
    b_eta = np.empty(n_btens)

    # Loop over b-tensor(s)
    for i in range(btens.shape[0]):
        i_btens = btens[i, :, :]
        i_bval, i_bdelta, i_b_eta = _btens_to_params_2d(i_btens, ztol)
        bval[i] = i_bval
        bdelta[i] = i_bdelta
        b_eta[i] = i_b_eta

    return bval, bdelta, b_eta


def params_to_btens(bval, bdelta, b_eta):
    """Compute b-tensor from trace, anisotropy and asymmetry parameters.

    Parameters
    ----------
    bval: int or float
        b-value (>= 0)
    bdelta: int or float
        normalized tensor anisotropy (>= -0.5 and <= 1)
    b_eta: int or float
        tensor asymmetry (>= 0 and <= 1)

    Returns
    -------
    (3, 3) numpy.ndarray
        output b-tensor

    Notes
    -----
    Implements eq. 7.11. p. 231 in [1].

    References
    ----------
    .. [1] D. Topgaard, NMR methods for studying microscopic diffusion
    anisotropy, in: R. Valiullin (Ed.), Diffusion NMR of Confined Systems: Fluid
    Transport in Porous Solids and Heterogeneous Materials, Royal Society of
    Chemistry, Cambridge, UK, 2016.

    """

    # Check input times are OK
    expected_input_types = (float, int)
    input_types_all_ok = (isinstance(bval, expected_input_types) and
                          isinstance(bdelta, expected_input_types) and
                          isinstance(b_eta, expected_input_types))

    if not input_types_all_ok:
        s = [x.__name__ for x in expected_input_types]
        it_msg = "All input types should any of: {}".format(s)
        raise ValueError(it_msg)

    # Check input values within expected ranges
    if bval < 0:
        raise ValueError("`bval` must be >= 0")

    if not -0.5 <= bdelta <= 1:
        raise ValueError("`delta` must be >= -0.5 and <= 1")

    if not 0 <= b_eta <= 1:
        raise ValueError("`b_eta` must be >= 0 and <= 1")

    m1 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]])
    m2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    btens = bval/3*(np.eye(3)+bdelta*(m1+b_eta*m2))

    return btens


def ornt_mapping(ornt1, ornt2):
    """Calculate the mapping needing to get from orn1 to orn2."""

    mapping = np.empty((len(ornt1), 2), 'int')
    mapping[:, 0] = -1
    A = ornt1[:, 0].argsort()
    B = ornt2[:, 0].argsort()
    mapping[B, 0] = A
    assert (mapping[:, 0] != -1).all()
    sign = ornt2[:, 1] * ornt1[mapping[:, 0], 1]
    mapping[:, 1] = sign
    return mapping


def reorient_vectors(bvecs, current_ornt, new_ornt, axis=0):
    """Change the orientation of gradients or other vectors.

    Moves vectors, storted along axis, from current_ornt to new_ornt. For
    example the vector [x, y, z] in "RAS" will be [-x, -y, z] in "LPS".

    R: Right
    A: Anterior
    S: Superior
    L: Left
    P: Posterior
    I: Inferior

    """
    if isinstance(current_ornt, str):
        current_ornt = orientation_from_string(current_ornt)
    if isinstance(new_ornt, str):
        new_ornt = orientation_from_string(new_ornt)

    n = bvecs.shape[axis]
    if current_ornt.shape != (n, 2) or new_ornt.shape != (n, 2):
        raise ValueError("orientations do not match")

    bvecs = np.asarray(bvecs)
    mapping = ornt_mapping(current_ornt, new_ornt)
    output = bvecs.take(mapping[:, 0], axis)
    out_view = np.rollaxis(output, axis, output.ndim)
    out_view *= mapping[:, 1]
    return output


def reorient_on_axis(bvecs, current_ornt, new_ornt, axis=0):
    if isinstance(current_ornt, str):
        current_ornt = orientation_from_string(current_ornt)
    if isinstance(new_ornt, str):
        new_ornt = orientation_from_string(new_ornt)

    n = bvecs.shape[axis]
    if current_ornt.shape != (n, 2) or new_ornt.shape != (n, 2):
        raise ValueError("orientations do not match")

    mapping = ornt_mapping(current_ornt, new_ornt)
    order = [slice(None)] * bvecs.ndim
    order[axis] = mapping[:, 0]
    shape = [1] * bvecs.ndim
    shape[axis] = -1
    sign = mapping[:, 1]
    sign.shape = shape
    output = bvecs[order]
    output *= sign
    return output


def orientation_from_string(string_ornt):
    """Return an array representation of an ornt string."""
    orientation_dict = dict(r=(0, 1), l=(0, -1), a=(1, 1),
                            p=(1, -1), s=(2, 1), i=(2, -1))
    ornt = tuple(orientation_dict[ii] for ii in string_ornt.lower())
    ornt = np.array(ornt)
    if _check_ornt(ornt):
        msg = string_ornt + " does not seem to be a valid orientation string"
        raise ValueError(msg)
    return ornt


def orientation_to_string(ornt):
    """Return a string representation of a 3d ornt."""
    if _check_ornt(ornt):
        msg = repr(ornt) + " does not seem to be a valid orientation"
        raise ValueError(msg)
    orientation_dict = {(0, 1): 'r', (0, -1): 'l', (1, 1): 'a',
                        (1, -1): 'p', (2, 1): 's', (2, -1): 'i'}
    ornt_string = ''
    for ii in ornt:
        ornt_string += orientation_dict[(ii[0], ii[1])]
    return ornt_string


def _check_ornt(ornt):
    uniq = np.unique(ornt[:, 0])
    if len(uniq) != len(ornt):
        print(len(uniq))
        return True
    uniq = np.unique(ornt[:, 1])
    if tuple(uniq) not in {(-1, 1), (-1,), (1,)}:
        print(tuple(uniq))
        return True
