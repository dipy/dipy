import itertools

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

        .. math::

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
    B[:, 6] = b * b * bvec[:, 0] ** 4 / 6
    B[:, 7] = b * b * bvec[:, 1] ** 4 / 6
    B[:, 8] = b * b * bvec[:, 2] ** 4 / 6
    B[:, 9] = 4 * b * b * bvec[:, 0] ** 3 * bvec[:, 1] / 6
    B[:, 10] = 4 * b * b * bvec[:, 0] ** 3 * bvec[:, 2] / 6
    B[:, 11] = 4 * b * b * bvec[:, 1] ** 3 * bvec[:, 0] / 6
    B[:, 12] = 4 * b * b * bvec[:, 1] ** 3 * bvec[:, 2] / 6
    B[:, 13] = 4 * b * b * bvec[:, 2] ** 3 * bvec[:, 0] / 6
    B[:, 14] = 4 * b * b * bvec[:, 2] ** 3 * bvec[:, 1] / 6
    B[:, 15] = b * b * bvec[:, 0] ** 2 * bvec[:, 1] ** 2
    B[:, 16] = b * b * bvec[:, 0] ** 2 * bvec[:, 2] ** 2
    B[:, 17] = b * b * bvec[:, 1] ** 2 * bvec[:, 2] ** 2
    B[:, 18] = 2 * b * b * bvec[:, 0] ** 2 * bvec[:, 1] * bvec[:, 2]
    B[:, 19] = 2 * b * b * bvec[:, 1] ** 2 * bvec[:, 0] * bvec[:, 2]
    B[:, 20] = 2 * b * b * bvec[:, 2] ** 2 * bvec[:, 0] * bvec[:, 1]
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

    B[:, 0] = -b1 * (n1[:, 0] ** 2) - b2 * (n2[:, 0] ** 2)
    B[:, 1] = -2 * b1 * n1[:, 0] * n1[:, 1] - 2 * b2 * n2[:, 0] * n2[:, 1]
    B[:, 2] = -b1 * n1[:, 1] ** 2 - b2 * n2[:, 1] ** 2
    B[:, 3] = -2 * b1 * n1[:, 0] * n1[:, 2] - 2 * b2 * n2[:, 0] * n2[:, 2]
    B[:, 4] = -2 * b1 * n1[:, 1] * n1[:, 2] - 2 * b2 * n2[:, 1] * n2[:, 2]
    B[:, 5] = -b1 * n1[:, 2] ** 2 - b2 * n2[:, 2] ** 2
    B[:, 6] = (b1 * b1 * n1[:, 0] ** 4 + b2 * b2 * n2[:, 0] ** 4) / 6
    B[:, 7] = (b1 * b1 * n1[:, 1] ** 4 + b2 * b2 * n2[:, 1] ** 4) / 6
    B[:, 8] = (b1 * b1 * n1[:, 2] ** 4 + b2 * b2 * n2[:, 2] ** 4) / 6
    B[:, 9] = (
        2 * b1 * b1 * n1[:, 0] ** 3 * n1[:, 1] + 2 * b2 * b2 * n2[:, 0] ** 3 * n2[:, 1]
    ) / 3
    B[:, 10] = (
        2 * b1 * b1 * n1[:, 0] ** 3 * n1[:, 2] + 2 * b2 * b2 * n2[:, 0] ** 3 * n2[:, 2]
    ) / 3
    B[:, 11] = (
        2 * b1 * b1 * n1[:, 1] ** 3 * n1[:, 0] + 2 * b2 * b2 * n2[:, 1] ** 3 * n2[:, 0]
    ) / 3
    B[:, 12] = (
        2 * b1 * b1 * n1[:, 1] ** 3 * n1[:, 2] + 2 * b2 * b2 * n2[:, 1] ** 3 * n2[:, 2]
    ) / 3
    B[:, 13] = (
        2 * b1 * b1 * n1[:, 2] ** 3 * n1[:, 0] + 2 * b2 * b2 * n2[:, 2] ** 3 * n2[:, 0]
    ) / 3
    B[:, 14] = (
        2 * b1 * b1 * n1[:, 2] ** 3 * n1[:, 1] + 2 * b2 * b2 * n2[:, 2] ** 3 * n2[:, 1]
    ) / 3
    B[:, 15] = (
        b1 * b1 * n1[:, 0] ** 2 * n1[:, 1] ** 2
        + b2 * b2 * n2[:, 0] ** 2 * n2[:, 1] ** 2
    )
    B[:, 16] = (
        b1 * b1 * n1[:, 0] ** 2 * n1[:, 2] ** 2
        + b2 * b2 * n2[:, 0] ** 2 * n2[:, 2] ** 2
    )
    B[:, 17] = (
        b1 * b1 * n1[:, 1] ** 2 * n1[:, 2] ** 2
        + b2 * b2 * n2[:, 1] ** 2 * n2[:, 2] ** 2
    )
    B[:, 18] = (
        2 * b1 * b1 * n1[:, 0] ** 2 * n1[:, 1] * n1[:, 2]
        + 2 * b2 * b2 * n2[:, 0] ** 2 * n2[:, 1] * n2[:, 2]
    )
    B[:, 19] = (
        2 * b1 * b1 * n1[:, 1] ** 2 * n1[:, 0] * n1[:, 2]
        + 2 * b2 * b2 * n2[:, 1] ** 2 * n2[:, 0] * n2[:, 2]
    )
    B[:, 20] = (
        2 * b1 * b1 * n1[:, 2] ** 2 * n1[:, 0] * n1[:, 1]
        + 2 * b2 * b2 * n2[:, 2] ** 2 * n2[:, 0] * n2[:, 1]
    )
    B[:, 21] = b1 * n1[:, 0] ** 2 * b2 * n2[:, 0] ** 2
    B[:, 22] = b1 * n1[:, 1] ** 2 * b2 * n2[:, 1] ** 2
    B[:, 23] = b1 * n1[:, 2] ** 2 * b2 * n2[:, 2] ** 2
    B[:, 24] = (
        b1 * n1[:, 1] ** 2 * b2 * n2[:, 2] ** 2
        + b1 * n1[:, 2] ** 2 * b2 * n2[:, 1] ** 2
    )
    B[:, 25] = (
        b1 * n1[:, 0] ** 2 * b2 * n2[:, 2] ** 2
        + b1 * n1[:, 2] ** 2 * b2 * n2[:, 0] ** 2
    )
    B[:, 26] = (
        b1 * n1[:, 0] ** 2 * b2 * n2[:, 1] ** 2
        + b1 * n1[:, 1] ** 2 * b2 * n2[:, 0] ** 2
    )
    B[:, 27] = 2 * (
        b1 * n1[:, 0] ** 2 * b2 * n2[:, 1] * n2[:, 2]
        + b1 * n1[:, 1] * n1[:, 2] * b2 * n2[:, 0] ** 2
    )
    B[:, 28] = 2 * (
        b1 * n1[:, 0] ** 2 * b2 * n2[:, 0] * n2[:, 2]
        + b1 * n1[:, 0] * n1[:, 2] * b2 * n2[:, 0] ** 2
    )
    B[:, 29] = 2 * (
        b1 * n1[:, 0] ** 2 * b2 * n2[:, 0] * n2[:, 1]
        + b1 * n1[:, 0] * n1[:, 1] * b2 * n2[:, 0] ** 2
    )
    B[:, 30] = 2 * (
        b1 * n1[:, 1] ** 2 * b2 * n2[:, 1] * n2[:, 2]
        + b1 * n1[:, 1] * n1[:, 2] * b2 * n2[:, 1] ** 2
    )
    B[:, 31] = 2 * (
        b1 * n1[:, 1] ** 2 * b2 * n2[:, 0] * n2[:, 2]
        + b1 * n1[:, 0] * n1[:, 2] * b2 * n2[:, 1] ** 2
    )
    B[:, 32] = 2 * (
        b1 * n1[:, 1] ** 2 * b2 * n2[:, 1] * n2[:, 0]
        + b1 * n1[:, 1] * n1[:, 0] * b2 * n2[:, 1] ** 2
    )
    B[:, 33] = 2 * (
        b1 * n1[:, 2] ** 2 * b2 * n2[:, 2] * n2[:, 1]
        + b1 * n1[:, 2] * n1[:, 1] * b2 * n2[:, 2] ** 2
    )
    B[:, 34] = 2 * (
        b1 * n1[:, 2] ** 2 * b2 * n2[:, 2] * n2[:, 0]
        + b1 * n1[:, 2] * n1[:, 0] * b2 * n2[:, 2] ** 2
    )
    B[:, 35] = 2 * (
        b1 * n1[:, 2] ** 2 * b2 * n2[:, 0] * n2[:, 1]
        + b1 * n1[:, 0] * n1[:, 1] * b2 * n2[:, 2] ** 2
    )
    B[:, 36] = 4 * (b1 * n1[:, 1] * n1[:, 2] * b2 * n2[:, 1] * n2[:, 2])
    B[:, 37] = 4 * (b1 * n1[:, 0] * n1[:, 2] * b2 * n2[:, 0] * n2[:, 2])
    B[:, 38] = 4 * (b1 * n1[:, 0] * n1[:, 1] * b2 * n2[:, 0] * n2[:, 1])
    B[:, 39] = 4 * (
        b1 * n1[:, 0] * n1[:, 2] * b2 * n2[:, 1] * n2[:, 2]
        + b1 * n1[:, 1] * n1[:, 2] * b2 * n2[:, 0] * n2[:, 2]
    )
    B[:, 40] = 4 * (
        b1 * n1[:, 0] * n1[:, 1] * b2 * n2[:, 0] * n2[:, 2]
        + b1 * n1[:, 0] * n1[:, 2] * b2 * n2[:, 0] * n2[:, 1]
    )
    B[:, 41] = 4 * (
        b1 * n1[:, 0] * n1[:, 1] * b2 * n2[:, 1] * n2[:, 2]
        + b1 * n1[:, 1] * n1[:, 2] * b2 * n2[:, 0] * n2[:, 1]
    )
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
            roi_radii[i] = min(int(roi_center[i]), int(data_shape[i] - roi_center[i]))
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


def convert_tensors(tensor, from_format, to_format):
    """Convert tensors from one format to another.

    Parameters
    ----------
    tensor : ndarray
        Input tensor.
    from_format : str
        Format of the input tensor. Options: 'dipy', 'mrtrix', 'ants', 'fsl'.
    to_format : str
        Format of the output tensor. Options: 'dipy', 'mrtrix', 'ants', 'fsl'.

    Notes
    -----
    - DIPY order: [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz].
      Shape: [i, j , k, 6].
      See: https://github.com/dipy/dipy/blob/master/dipy/reconst/dti.py#L1639
    - MRTRIX order: [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
       Shape: [i, j , k, 6].
       See: https://mrtrix.readthedocs.io/en/dev/reference/commands/dwi2tensor.html  # noqa
    - ANTS: [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz].
       Shape: [i, j , k, 1, 6]  -  Note the extra dimension (5D)
       See: https://github.com/ANTsX/ANTs/wiki/Importing-diffusion-tensor-data-from-other-software  # noqa
    - FSL: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
      Shape: [i, j , k, 6]. (Also used for the Fibernavigator)
      Ref: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide
    """  # noqa: E501
    tensor_order = {
        "fsl": [[0, 1, 3, 2, 4, 5], [0, 1, 3, 2, 4, 5]],
        "mrtrix": [[0, 3, 1, 4, 5, 2], [0, 2, 5, 1, 3, 4]],
        "dipy": [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
        "ants": [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
    }

    if from_format.lower() not in tensor_order.keys():
        raise ValueError(f"Unknown tensor format: {from_format}")
    if to_format.lower() not in tensor_order.keys():
        raise ValueError(f"Unknown tensor format: {to_format}")

    if from_format.lower() == to_format.lower():
        return tensor

    if from_format.lower() in ["ants", "dipy"]:
        tensor = np.squeeze(tensor) if tensor.ndim == 5 else tensor

    tensor_dipy = tensor[..., tensor_order[from_format.lower()][0]]

    if to_format.lower() == "ants":
        tensor_dipy = tensor_dipy[:, :, :, np.newaxis, :]
        return tensor_dipy
    elif to_format.lower() == "dipy":
        return tensor_dipy

    tensor_reordered = tensor_dipy[..., tensor_order[to_format.lower()][1]]
    return tensor_reordered


def compute_coherence_table_for_gradient_transforms(
    fiber_orientations,
    anisotropy_values,
    *,
    nb_flips=4,
    angle_threshold=None,
):
    """Compute fiber coherence indexes for gradient table orientation variations.

    This function explores potential gradient table orientation errors by
    computing coherence across 24 possible permutations and flips of the
    original gradient directions. See :footcite:p:`schilling2019b`.

    Parameters
    ----------
    fiber_orientations : np.ndarray
        Principal fiber orientations (e.g peaks) for each voxel, shape (..., 3)
    anisotropy_values : np.ndarray
        Anisotropy measure for each voxel (e.g. FA map), shape should match
        fiber_orientations.shape[:-1]
    nb_flips : int, optional
        Number of flips to consider for each permutation.
    angle_threshold : float, optional
        Maximum angle for considering directions coherent.

    Returns
    -------
    Tuple containing:
    - List of coherence values for each transform
    - List of transformation matrices

    References
    ----------
    .. footbibliography::

    """
    # Precompute all possible permutations and flips
    permutations = list(itertools.permutations([0, 1, 2]))
    nb_transforms = len(permutations) * nb_flips
    transforms = np.zeros((nb_transforms, 3, 3), dtype=np.float32)

    for i, perm in enumerate(permutations):
        # Base permutation transform
        transforms[i * nb_flips][np.arange(3), perm] = 1.0

        # Flip transforms for each axis
        for axis in range(3):
            flip_transform = transforms[i * nb_flips].copy()
            flip_transform[axis, axis] *= -1
            transforms[i * nb_flips + axis + 1] = flip_transform

    ndims = len(fiber_orientations.shape) - 1
    neighbor_offsets = np.array(
        [
            offset
            for offset in itertools.product([-1, 0, 1], repeat=ndims)
            if any(offset)  # Exclude (0,0,0)
        ]
    )
    # Compute coherence for each transform
    coherence_values = [
        compute_fiber_coherence(
            np.einsum("...ij,...j->...i", transform, fiber_orientations),
            anisotropy_values,
            angle_threshold=angle_threshold,
            neighbor_offsets=neighbor_offsets,
        )
        for transform in transforms
    ]

    return coherence_values, list(transforms)


def compute_fiber_coherence(
    peaks,
    anisotropy_values,
    *,
    angle_threshold=None,
    neighbor_offsets=None,
):
    """Compute fiber coherence by analyzing directional consistency.

    See :footcite:p:`schilling2019b` for more details.

    Parameters
    ----------
    peaks : np.ndarray
        Principal fiber orientations for each voxel, shape (..., 3)
    anisotropy_values : np.ndarray
        Anisotropy measure for each voxel (e.g. FA map).
        Shape should match peaks.shape[:-1]
    angle_threshold : float, optional
        Maximum angle for considering directions coherent in radians.
    neighbor_offsets : np.ndarray, optional
        Precomputed neighbor offsets for efficient computation

    Returns
    -------
    float
        Computed fiber coherence value
        Higher values indicate more coherent fiber orientations

    References
    ----------
    .. footbibliography::

    """
    if peaks.shape[:-1] != anisotropy_values.shape:
        raise ValueError(
            f"Shape mismatch: peaks {peaks.shape}, values {anisotropy_values.shape}"
        )

    angle_threshold = angle_threshold or np.pi / 6.0
    with np.errstate(divide="ignore", invalid="ignore"):
        norms = np.linalg.norm(peaks, axis=-1, keepdims=True)
        norm_peaks = np.divide(peaks, norms, out=np.zeros_like(peaks), where=norms > 0)

    ndims = len(peaks.shape) - 1
    if neighbor_offsets is None:
        # Compute neighbor offsets and exclude zero offset
        neighbor_offsets = np.array(
            [
                offset
                for offset in itertools.product([-1, 0, 1], repeat=ndims)
                if any(offset)  # Exclude (0,0,0)
            ]
        )

    coherence = 0.0
    cos_threshold = np.cos(angle_threshold)
    for offset in neighbor_offsets:
        # Create shifted slices to align neighborhoods
        slice_center = tuple(slice(1, -1) for _ in range(ndims))
        slice_neighbor = tuple(
            slice(1 + off, -1 + off if off <= 0 else None) for off in offset
        )

        peaks_center = norm_peaks[slice_center]
        peaks_neighbor = norm_peaks[slice_neighbor]
        # Compute absolute dot product (cos of angle)
        # Using einsum for efficient dot product calculation
        cos_angles = np.abs(np.einsum("...i,...i->...", peaks_center, peaks_neighbor))

        # Find voxels with angle less than threshold
        coherent_mask = cos_angles > cos_threshold

        values_center = anisotropy_values[slice_center]
        values_neighbor = anisotropy_values[slice_neighbor]

        # Add to coherence score
        # Only count voxels where both center and neighbor have significant values
        valid_mask = (values_center > 0) & (values_neighbor > 0) & coherent_mask
        coherence += np.sum(values_center[valid_mask] + values_neighbor[valid_mask])

    return coherence
