import numpy as np
import pytest

from dipy.core.geometry import normalized_vector
from dipy.core.gradients import gradient_table
from dipy.stats.qc import (
    dwi_contrast,
    find_qspace_contrast,
    neighboring_dwi_correlation,
)

rng = np.random.default_rng()


def create_test_data(test_r, cube_size, mask_size, num_dwi_vols, num_b0s):
    """Create testing data with a known neighbor structure and a known NDC.

    The b>0 images have 2 images per shell, separated by a very small angle,
    guaranteeing they will be neighbors. The within-mask data is filled with
    random data with correlation value of approximately ``test_r``.

    Parameters
    ----------

    test_r: float
        The approximate NDC that the simulated data should have
    cube_size: int
        The simulated data will be a cube with this many voxels per dim
    mask_size: int
        A cubic "brain" is this size per side and filled with data. Must
        be less than ``cube_size``
    num_dwi_vols: int
        The number of b>0 images to simulate. Must be even to ensure we
        can make known neighbors
    num_b0s: int
        The number of b=0 images to prepend to the b>0 images

    Returns
    -------

    real_r: float
        The ground-truth neighbor correlation of the simulated data
    dwi_data: np.ndarray
        A 4D array containing simulated data
    mask_data: np.ndarray
        A 3D array indicating which voxels in ``dwi_data`` contain
        brain data
    gtab: dipy.core.gradients.GradientTable
        Gradient table with known neighbors

    """

    if not num_dwi_vols % 2 == 0:
        raise Exception("Needs an even number of dwi vols to ensure known neighbors")

    # Create a volume mask
    test_mask = np.zeros((cube_size, cube_size, cube_size))
    test_mask[:mask_size, :mask_size, :mask_size] = 1
    n_voxels_in_mask = mask_size**3

    # 4D Data array
    dwi_data = np.zeros((cube_size, cube_size, cube_size, num_b0s + num_dwi_vols))

    # Create a sampling scheme where we know what volumes will be neighbors
    n_known = num_dwi_vols // 2
    dwi_bvals = (
        np.column_stack([np.arange(n_known) + 1] * 2).flatten(order="C").tolist()
    )
    bvals = np.array([0] * num_b0s + dwi_bvals) * 1000

    # The bvecs will be a straight line with a minor perturbance every other
    ref_vec = np.array([1.0, 0.0, 0.0])
    nbr_vec = normalized_vector(ref_vec + 0.00001)
    bvecs = np.vstack([ref_vec] * num_b0s + [np.vstack([ref_vec, nbr_vec])] * n_known)

    cor = np.ones((2, 2)) * test_r
    np.fill_diagonal(cor, 1)
    L = np.linalg.cholesky(cor)

    known_correlations = []
    for starting_vol in np.arange(n_known) * 2 + num_b0s:
        uncorrelated = rng.standard_normal((2, n_voxels_in_mask))
        correlated = np.dot(L, uncorrelated)

        dwi_data[:, :, :, starting_vol][test_mask > 0] = correlated[0]
        dwi_data[:, :, :, starting_vol + 1][test_mask > 0] = correlated[1]

        known_correlations += [np.corrcoef(correlated)[0, 1]] * 2

    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)

    return np.mean(known_correlations), dwi_data, test_mask, gtab


def test_neighboring_dwi_correlation():
    """Test NDC under various conditions."""

    # Test data with b=0s, low correlation, using mask
    real_r, dwi_data, mask, gtab = create_test_data(
        test_r=0.3, cube_size=10, mask_size=6, num_dwi_vols=10, num_b0s=2
    )
    estimated_ndc = neighboring_dwi_correlation(dwi_data, gtab, mask=mask)
    assert np.allclose(real_r, estimated_ndc)

    maskless_ndc = neighboring_dwi_correlation(dwi_data, gtab)
    assert maskless_ndc != real_r

    # Try with no b=0s
    real_r, dwi_data, mask, gtab = create_test_data(
        test_r=0.3, cube_size=10, mask_size=6, num_dwi_vols=10, num_b0s=0
    )
    estimated_ndc = neighboring_dwi_correlation(dwi_data, gtab, mask=mask)
    assert np.allclose(real_r, estimated_ndc)

    # Try with realistic correlation value
    real_r, dwi_data, mask, gtab = create_test_data(
        test_r=0.8, cube_size=10, mask_size=6, num_dwi_vols=10, num_b0s=2
    )
    estimated_ndc = neighboring_dwi_correlation(dwi_data, gtab, mask=mask)
    assert np.allclose(real_r, estimated_ndc)

    # Try with a bigger volume, lower correlation
    real_r, dwi_data, mask, gtab = create_test_data(
        test_r=0.5, cube_size=100, mask_size=49, num_dwi_vols=160, num_b0s=2
    )
    estimated_ndc = neighboring_dwi_correlation(dwi_data, gtab, mask=mask)
    assert np.allclose(real_r, estimated_ndc)


def create_contrast_test_data(num_b0s=1):
    """Create DWI data with known neighbor and contrast correlations.

    Four DWI volumes are used. The first two are nearly parallel to the
    x-axis and the second two are nearly parallel to the y-axis. Thus,
    volumes within each pair are q-space neighbors, while volumes from
    the other pair provide the contrast directions.

    The image data are constructed to have an exact known correlation
    matrix.

    Parameters
    ----------
    num_b0s : int, optional
        Number of b=0 volumes to prepend.

    Returns
    -------
    expected_contrast : float
        Expected DWI contrast.
    dwi_data : ndarray
        Simulated 4D DWI data.
    mask : ndarray
        Mask containing all simulated brain voxels.
    gtab : dipy.core.gradients.GradientTable
        Gradient table with known neighbor and contrast relationships.
    """
    # Eight voxels are enough to construct four mutually orthogonal
    # zero-mean basis signals.
    mask = np.ones((2, 2, 2), dtype=bool)

    basis = np.array(
        [
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, -1, -1, 1, -1, 1, 1, -1],
        ],
        dtype=float,
    )

    # Correlations between the four DWI volumes.
    #
    # Neighbor pairs:
    #   0 <-> 1 : 0.8
    #   2 <-> 3 : 0.8
    #
    # Contrast selections will be:
    #   0 -> 2 : 0.20
    #   1 -> 2 : 0.16
    #   2 -> 0 : 0.20
    #   3 -> 0 : 0.50
    correlation = np.array(
        [
            [1.00, 0.80, 0.20, 0.50],
            [0.80, 1.00, 0.16, 0.40],
            [0.20, 0.16, 1.00, 0.80],
            [0.50, 0.40, 0.80, 1.00],
        ]
    )

    # Because the basis signals are mutually orthogonal and have equal
    # variance, applying the Cholesky factor produces signals having the
    # correlation matrix above.
    L = np.linalg.cholesky(correlation)
    signals = L @ basis

    dwi_data = np.zeros((2, 2, 2, num_b0s + 4))

    for index in range(4):
        dwi_data[..., num_b0s + index][mask] = signals[index]

    # Two very close directions around x and two around y.
    x = np.array([1.0, 0.0, 0.0])
    x_neighbor = normalized_vector(np.array([1.0, 0.001, 0.0]))
    y = np.array([0.0, 1.0, 0.0])
    y_neighbor = normalized_vector(np.array([0.001, 1.0, 0.0]))

    bvals = np.array([0] * num_b0s + [1000] * 4)

    bvecs = np.vstack(
        [
            np.zeros((num_b0s, 3)),
            x,
            x_neighbor,
            y,
            y_neighbor,
        ]
    )

    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)

    neighbor_correlations = [
        correlation[0, 1],
        correlation[1, 0],
        correlation[2, 3],
        correlation[3, 2],
    ]

    contrast_correlations = [
        correlation[0, 2],
        correlation[1, 2],
        correlation[2, 0],
        correlation[3, 0],
    ]

    expected_contrast = np.mean(neighbor_correlations) / np.mean(contrast_correlations)

    return expected_contrast, dwi_data, mask, gtab


def test_find_qspace_contrast():
    """Test that the expected contrast DWI is selected."""

    _, _, _, gtab = create_contrast_test_data(num_b0s=1)

    contrast_indices = find_qspace_contrast(gtab)

    # Volume layout:
    # 0: b0
    # 1: x
    # 2: x + small perturbation
    # 3: y
    # 4: y + small perturbation
    #
    # The x-like volumes should select y as their contrast DWI,
    # and the y-like volumes should select x.
    expected_indices = [
        (1, 3),
        (2, 3),
        (3, 1),
        (4, 1),
    ]

    assert contrast_indices == expected_indices


def test_dwi_contrast():
    """Test DWI contrast with known correlations."""

    expected_contrast, dwi_data, mask, gtab = create_contrast_test_data(num_b0s=1)

    estimated_contrast = dwi_contrast(
        dwi_data,
        gtab,
        mask=mask,
    )

    assert np.allclose(expected_contrast, estimated_contrast)

    # Since all voxels are included in the mask, the result should be
    # identical without explicitly providing the mask.
    estimated_contrast_no_mask = dwi_contrast(
        dwi_data,
        gtab,
    )

    assert np.allclose(
        expected_contrast,
        estimated_contrast_no_mask,
    )


def test_find_qspace_contrast_multishell():
    """Test contrast DWI selection with different q-space magnitudes."""

    # Reference direction at b=1000.
    #
    # Volume 1 is exactly orthogonal, but at b=4000.
    # Volume 2 is not exactly orthogonal (60 degrees), but is at the
    # same b-value as the reference.
    #
    # Yeh's normalized perpendicular-vector criterion should select
    # volume 2 for volume 0.
    bvals = np.array([1000, 4000, 1000])

    bvecs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
        ]
    )

    gtab = gradient_table(
        bvals,
        bvecs=bvecs,
        b0_threshold=50,
    )

    contrast_indices = find_qspace_contrast(gtab)

    assert contrast_indices[0] == (0, 2)


def test_find_qspace_contrast_nearly_parallel():
    """Test that nearly parallel vectors are treated as parallel."""

    bvals = np.array([1000, 1000])

    bvecs = np.array(
        [
            [1.0, 0.0, 0.0],
            normalized_vector(np.array([1.0, 1e-12, 0.0])),
        ]
    )

    gtab = gradient_table(
        bvals,
        bvecs=bvecs,
        b0_threshold=50,
    )

    with pytest.raises(
        ValueError,
        match="At least one non-parallel DWI direction is required",
    ):
        find_qspace_contrast(gtab)
