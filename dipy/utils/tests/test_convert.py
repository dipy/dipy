"""Testing convert utilities."""

import numpy as np

from dipy.utils.convert import expand_range


def test_expand_range():
    test_cases = [
        ("1,2,3,4", np.array([1, 2, 3, 4])),
        ("5-7", np.array([5, 6, 7])),
        ("0", np.array([0])),
        ("0,", np.array([0])),
        ("0, ", np.array([0])),
        ("1,2,5-7,10", np.array([1, 2, 5, 6, 7, 10])),
        ("1,2,5-7,10", [1, 2, 5, 6, 7, 10]),
        ("1,2,5-7,10", (1, 2, 5, 6, 7, 10)),
        ("1,a,3", ValueError),
        ("1-2-3", ValueError)
    ]

    for input_string, expected_output in test_cases:
        # print(f"Running test for input: {input_string}")
        if not isinstance(expected_output, (np.ndarray, list, tuple)):
            try:
                expand_range(input_string)
                assert False, f"Expected ValueError for input: {input_string}"
            except ValueError:
                assert True
        else:
            result = expand_range(input_string)
            np.testing.assert_array_equal(result, expected_output)
            # print(f"Test passed for input: {input_string}")
