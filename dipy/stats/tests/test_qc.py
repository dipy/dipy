import numpy as np
from dipy.stats.qc import neighboring_dwi_correlation
from dipy.core.gradients import gradient_table
from dipy.core.geometry import normalized_vector

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
        raise Exception(
            "Needs an even number of dwi vols to ensure known neighbors")

    # Create a volume mask
    test_mask = np.zeros((cube_size, cube_size, cube_size))
    test_mask[:mask_size, :mask_size, :mask_size] = 1
    n_voxels_in_mask = mask_size ** 3

    # 4D Data array
    dwi_data = np.zeros(
        (cube_size, cube_size, cube_size, num_b0s + num_dwi_vols))

    # Create a sampling scheme where we know what volumes will be neighbors
    n_known = num_dwi_vols // 2
    dwi_bvals = np.column_stack(
        [np.arange(n_known) + 1] * 2).flatten(order="C").tolist()
    bvals = np.array([0] * num_b0s + dwi_bvals) * 1000

    # The bvecs will be a straight line with a minor perturbance every other
    ref_vec = np.array([1., 0., 0.])
    nbr_vec = normalized_vector(ref_vec + 0.00001)
    bvecs = np.row_stack(
        [ref_vec] * num_b0s + [np.row_stack([ref_vec, nbr_vec])] * n_known)

    cor = np.ones((2, 2)) * test_r
    np.fill_diagonal(cor, 1)
    L = np.linalg.cholesky(cor)

    known_correlations = []
    for starting_vol in np.arange(n_known) * 2 + num_b0s:
        uncorrelated = rng.standard_normal((2, n_voxels_in_mask))
        correlated = np.dot(L, uncorrelated)

        dwi_data[:, :, :, starting_vol][test_mask > 0] = correlated[0]
        dwi_data[:, :, :, starting_vol+1][test_mask > 0] = correlated[1]

        known_correlations += [np.corrcoef(correlated)[0, 1]] * 2

    gtab = gradient_table(bvals, bvecs, b0_threshold=50)

    return np.mean(known_correlations), dwi_data, test_mask, gtab


def test_neighboring_dwi_correlation():
    """Test NDC under various conditions."""

    # Test data with b=0s, low correlation, using mask
    real_r, dwi_data, mask, gtab = create_test_data(
        test_r=0.3,
        cube_size=10,
        mask_size=6,
        num_dwi_vols=10,
        num_b0s=2)
    estimated_ndc = neighboring_dwi_correlation(dwi_data, gtab, mask)
    assert np.allclose(real_r, estimated_ndc)

    maskless_ndc = neighboring_dwi_correlation(dwi_data, gtab)
    assert maskless_ndc != real_r

    # Try with no b=0s
    real_r, dwi_data, mask, gtab = create_test_data(
        test_r=0.3,
        cube_size=10,
        mask_size=6,
        num_dwi_vols=10,
        num_b0s=0)
    estimated_ndc = neighboring_dwi_correlation(dwi_data, gtab, mask)
    assert np.allclose(real_r, estimated_ndc)

    # Try with realistic correlation value
    real_r, dwi_data, mask, gtab = create_test_data(
        test_r=0.8,
        cube_size=10,
        mask_size=6,
        num_dwi_vols=10,
        num_b0s=2)
    estimated_ndc = neighboring_dwi_correlation(dwi_data, gtab, mask)
    assert np.allclose(real_r, estimated_ndc)

    # Try with a bigger volume, lower correlation
    real_r, dwi_data, mask, gtab = create_test_data(
        test_r=0.5,
        cube_size=100,
        mask_size=49,
        num_dwi_vols=160,
        num_b0s=2)
    estimated_ndc = neighboring_dwi_correlation(dwi_data, gtab, mask)
    assert np.allclose(real_r, estimated_ndc)
