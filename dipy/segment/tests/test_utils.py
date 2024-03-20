import warnings

import numpy as np
from dipy.segment.utils import remove_holes_and_islands
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator()
def test_remove_holes_and_islands(rng=None):
    # inefficient but more readable
    temp = rng.integers(2, size=(40, 40, 40))
    temp[9:31, 9:31, 9:31] = 0
    temp[10:30, 10:30, 10:30] = 1
    temp[11:29, 11:29, 11:29] = rng.integers(2, size=(18, 18, 18))
    output = remove_holes_and_islands(temp)
    ground_truth = np.zeros((40, 40, 40))
    ground_truth[10:30, 10:30, 10:30] = 1
    np.testing.assert_equal(output, ground_truth)
