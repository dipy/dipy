"""Testing utilities."""

import numpy as np

from dipy.workflows.utils import handle_vol_idx


def test_handle_vol_idx():
    test_cases = [
        ("1,2,5-7,10", np.array([1, 2, 5, 6, 7, 10])),
        (3, [3]),
        ([3, '2', 1], [3, 2, 1])
    ]

    for input_val, expected_output in test_cases:
        np.testing.assert_array_equal(
            handle_vol_idx(input_val), expected_output)
