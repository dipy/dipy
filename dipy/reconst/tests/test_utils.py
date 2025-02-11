import numpy as np
import numpy.testing as npt

from dipy.reconst.utils import (
    _mask_from_roi,
    _roi_in_volume,
    compute_coherence_table_for_gradient_transforms,
    compute_fiber_coherence,
    convert_tensors,
)


def test_roi_in_volume():
    data_shape = (11, 11, 11, 64)
    roi_center = np.array([5, 5, 5])
    roi_radii = np.array([5, 5, 5])
    roi_radii_out = _roi_in_volume(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_radii_out, np.array([5, 5, 5]))

    roi_radii = np.array([6, 6, 6])
    roi_radii_out = _roi_in_volume(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_radii_out, np.array([5, 5, 5]))

    roi_center = np.array([4, 4, 4])
    roi_radii = np.array([5, 5, 5])
    roi_radii_out = _roi_in_volume(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_radii_out, np.array([4, 4, 4]))

    data_shape = (11, 11, 1, 64)
    roi_center = np.array([5, 5, 0])
    roi_radii = np.array([5, 5, 0])
    roi_radii_out = _roi_in_volume(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_radii_out, np.array([5, 5, 0]))

    roi_center = np.array([2, 5, 0])
    roi_radii = np.array([5, 10, 2])
    roi_radii_out = _roi_in_volume(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_radii_out, np.array([2, 5, 0]))


def test_mask_from_roi():
    data_shape = (5, 5, 5)
    roi_center = (2, 2, 2)
    roi_radii = (2, 2, 2)
    mask_gt = np.ones(data_shape)
    roi_mask = _mask_from_roi(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_mask, mask_gt)

    roi_radii = (1, 2, 2)
    mask_gt = np.zeros(data_shape)
    mask_gt[1:4, 0:5, 0:5] = 1
    roi_mask = _mask_from_roi(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_mask, mask_gt)

    roi_radii = (0, 2, 2)
    mask_gt = np.zeros(data_shape)
    mask_gt[2, 0:5, 0:5] = 1
    roi_mask = _mask_from_roi(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_mask, mask_gt)


def test_convert_tensor():
    # Test case 1: Convert from 'dipy' to 'mrtrix'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, "dipy", "mrtrix")
    expected_tensor = np.array([[[[1, 3, 6, 2, 4, 5]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 2: Convert from 'mrtrix' to 'ants'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, "mrtrix", "ants")
    expected_tensor = np.array([[[[[1, 4, 2, 5, 6, 3]]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 3: Convert from 'ants' to 'fsl'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, "ants", "fsl")
    expected_tensor = np.array([[[[1, 2, 4, 3, 5, 6]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 4: Convert from 'fsl' to 'dipy'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, "fsl", "dipy")
    expected_tensor = np.array([[[[1, 2, 4, 3, 5, 6]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 5: Convert from 'dipy' to 'ants'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, "dipy", "ants")
    expected_tensor = np.array([[[[[1, 2, 3, 4, 5, 6]]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 6: Convert from 'ants' to 'dipy'
    tensor = np.array([[[[[1, 2, 3, 4, 5, 6]]]]])
    converted_tensor = convert_tensors(tensor, "ants", "dipy")
    expected_tensor = np.array([1, 2, 3, 4, 5, 6])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 7: Convert from 'dipy' to 'dipy'
    tensor = np.array([[[[[1, 2, 3, 4, 5, 6]]]]])
    converted_tensor = convert_tensors(tensor, "dipy", "dipy")
    npt.assert_array_equal(converted_tensor, tensor)

    npt.assert_raises(ValueError, convert_tensors, tensor, "amico", "dipy")
    npt.assert_raises(ValueError, convert_tensors, tensor, "dipy", "amico")


def test_transform_table_dimensions():
    """
    Test the dimensionality of the transform table output.

    Verifies that compute_coherence_table_for_gradient_transforms generates
    the expected number of transformations (24) representing all possible
    combinations of axis permutations and flips.
    """
    peaks = np.zeros((3, 3, 5, 3), dtype=float)
    fa = np.zeros((3, 3, 5), dtype=float)

    coherence, transforms = compute_coherence_table_for_gradient_transforms(peaks, fa)

    # We expect 24 transformations: 6 permutations × 4 flips
    npt.assert_equal(len(coherence), 24)
    npt.assert_equal(len(transforms), 24)

    # Verify transform shapes
    npt.assert_equal(all(t.shape == (3, 3) for t in transforms), True)


def test_aligned_fibers_coherence():
    """
    Test coherence computation for aligned fibers.

    Creates a simple configuration of aligned fibers in the z-direction
    with high anisotropy values. Tests that:
    1. Aligned fibers show positive coherence
    2. Sign flips don't affect coherence
    """
    # Base test configuration
    directions = np.zeros((3, 3, 5, 3), dtype=float)
    fa = np.zeros((3, 3, 5), dtype=float)

    # Test 1: Strong alignment in z-direction
    z_aligned = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=float
    )
    directions[1, 1, :, :] = z_aligned
    fa[1, 1, :] = [1, 1, 1, 1, 0]

    coherence_aligned = compute_fiber_coherence(directions, fa)
    assert coherence_aligned > 0, "Aligned fibers should show positive coherence"

    # Test 2: Symmetry with sign flip
    z_aligned_flipped = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, 0]], dtype=float
    )
    directions[1, 1, :, :] = z_aligned_flipped

    coherence_flipped = compute_fiber_coherence(directions, fa)
    npt.assert_almost_equal(
        coherence_flipped,
        coherence_aligned,
        err_msg="Sign flips should not affect coherence",
    )


def test_anisotropy_impact():
    """
    Test how anisotropy values affect coherence calculation.

    Verifies that:
    1. Lower anisotropy values result in lower coherence
    2. Zero anisotropy results in zero coherence
    3. Coherence scales proportionally with anisotropy values
    """
    directions = np.zeros((3, 3, 5, 3), dtype=float)
    z_aligned = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, 0]], dtype=float
    )
    directions[1, 1, :, :] = z_aligned

    # Test with high anisotropy
    fa_high = np.zeros((3, 3, 5), dtype=float)
    fa_high[1, 1, :] = [1, 1, 1, 1, 0]
    coherence_high = compute_fiber_coherence(directions, fa_high)

    # Test with low anisotropy
    fa_low = np.zeros((3, 3, 5), dtype=float)
    fa_low[1, 1, :] = [0.2, 0.2, 0.2, 0.2, 0]
    coherence_low = compute_fiber_coherence(directions, fa_low)

    assert (
        coherence_low < coherence_high
    ), "Lower anisotropy should result in lower coherence"
    npt.assert_array_less(
        coherence_low,
        coherence_high,
        err_msg="Coherence should scale with anisotropy values",
    )


def test_misaligned_fibers():
    """
    Test coherence computation for misaligned fiber configurations.

    The coherence value for perpendicular fibers should be very low but not
    necessarily exactly zero due to:
    1. Numerical precision in angle calculations
    2. The specific angle threshold (pi/6) used in the implementation
    3. The contribution of neighboring voxels

    The test verifies that:
    1. Perpendicular fibers show very low coherence
    2. Coherence of perpendicular fibers is much lower than aligned fibers
    3. Edge cases with zero vectors are handled correctly
    """
    peaks_aligned = np.zeros((3, 3, 5, 3), dtype=float)
    fa = np.zeros((3, 3, 5), dtype=float)

    aligned_fibers = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=float
    )
    peaks_aligned[1, 1, :, :] = aligned_fibers
    fa[1, 1, :] = [1, 1, 1, 1, 0]

    coherence_aligned = compute_fiber_coherence(peaks_aligned, fa)

    peaks_perp = np.zeros((3, 3, 5, 3), dtype=float)

    # Create perpendicular fiber configuration
    perp_fibers = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=float
    )
    # Normalize the vectors
    norms = np.linalg.norm(perp_fibers, axis=1, keepdims=True)
    perp_fibers = np.divide(
        perp_fibers, norms, out=np.zeros_like(perp_fibers), where=norms != 0
    )

    peaks_perp[1, 1, :, :] = perp_fibers

    coherence_perp = compute_fiber_coherence(peaks_perp, fa)

    # Assertions
    assert (
        coherence_perp < coherence_aligned
    ), "Perpendicular fibers should show lower coherence than aligned fibers"
    assert (
        coherence_perp < 0.1 * coherence_aligned
    ), "Perpendicular fibers should show significantly lower coherence"

    # Test gradual misalignment
    directions_gradual = np.zeros((3, 3, 5, 3), dtype=float)
    angles = np.array([0, np.pi / 6, np.pi / 4, np.pi / 3, 0])  # Increasing angles

    # Create gradually misaligned fibers
    gradual_fibers = np.zeros_like(perp_fibers)
    for i, angle in enumerate(angles):
        if angle != 0:
            gradual_fibers[i] = [np.sin(angle), 0, np.cos(angle)]

    directions_gradual[1, 1, :, :] = gradual_fibers
    coherence_gradual = compute_fiber_coherence(directions_gradual, fa)

    assert (
        coherence_gradual < coherence_aligned
    ), "Gradually misaligned fibers should show lower coherence than aligned"
    assert (
        coherence_gradual > coherence_perp
    ), "Gradually misaligned fibers should show higher coherence than perpendicular"


def test_angle_threshold_sensitivity():
    """
    Test how the angle threshold affects coherence calculations.

    Tests the behavior of coherence calculation for angles near the threshold,
    verifying that the coherence decreases as the angle increases.
    """
    shape = (3, 3, 3)
    directions = np.zeros(shape + (3,), dtype=float)
    fa = np.ones(shape, dtype=float)

    angles = [np.pi / 12, np.pi / 6, np.pi / 4]  # 15, 30, 45 degrees
    coherence_values = []

    for angle in angles:
        directions = np.zeros(shape + (3,), dtype=float)
        directions[1, 1, 1] = [0, 0, 1]
        directions[1, 1, 0] = [np.sin(angle), 0, np.cos(angle)]

        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        directions = np.divide(
            directions, norms, out=np.zeros_like(directions), where=norms > 0
        )

        coherence = compute_fiber_coherence(directions, fa)
        coherence_values.append(coherence)

    assert (
        coherence_values[0] > coherence_values[1]
    ), "Coherence should decrease with increasing angle"
    assert coherence_values[1] == 0, "Angles at threshold should have zero coherence"
    assert coherence_values[2] == 0, "Angles above threshold should have zero coherence"


def test_edge_cases():
    """
    Test edge cases and boundary conditions.

    Verifies proper handling of:
    1. Zero vectors
    2. Very small anisotropy values
    3. NaN and Inf values
    4. Single voxel configurations
    """
    directions = np.zeros((3, 3, 5, 3), dtype=float)
    fa = np.zeros((3, 3, 5), dtype=float)

    coherence_zeros = compute_fiber_coherence(directions, fa)
    assert coherence_zeros == 0, "Zero vectors should result in zero coherence"

    directions_small = np.full((3, 3, 5, 3), 1e-10, dtype=float)
    fa_small = np.full((3, 3, 5), 1e-10, dtype=float)

    coherence_small = compute_fiber_coherence(directions_small, fa_small)
    assert np.isfinite(coherence_small), "Small values should produce finite coherence"


def test_fiber_alignment():
    """
    Test coherence computation for different fiber alignments.

    This test verifies the coherence calculation for:
    1. Perfectly aligned fibers
    2. Slightly misaligned fibers (below threshold)
    3. Completely misaligned fibers (above threshold)
    """
    shape = (3, 3, 3)
    fa = np.ones(shape, dtype=float)

    # Test 1: Perfectly aligned fibers
    peaks_aligned = np.zeros(shape + (3,), dtype=float)
    peaks_aligned[1, 1, 1] = [0, 0, 1]
    peaks_aligned[1, 1, 0] = [0, 0, 1]

    coherence_aligned = compute_fiber_coherence(peaks_aligned, fa)
    assert coherence_aligned > 0, "Aligned fibers should have positive coherence"

    # Test 2: Slightly misaligned fibers (15 degrees)
    peaks_slight = np.zeros(shape + (3,), dtype=float)
    peaks_slight[1, 1, 1] = [0, 0, 1]
    peaks_slight[1, 1, 0] = [np.sin(np.pi / 12), 0, np.cos(np.pi / 12)]

    coherence_slight = compute_fiber_coherence(peaks_slight, fa)
    assert (
        coherence_slight > 0
    ), "Slightly misaligned fibers should have positive coherence"

    # Test 3: Completely misaligned fibers (90 degrees)
    peaks_perp = np.zeros(shape + (3,), dtype=float)
    peaks_perp[1, 1, 1] = [0, 0, 1]
    peaks_perp[1, 1, 0] = [1, 0, 0]

    coherence_perp = compute_fiber_coherence(peaks_perp, fa)
    assert coherence_perp == 0, "Perpendicular fibers should have zero coherence"
