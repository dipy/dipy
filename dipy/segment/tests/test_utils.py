import numpy as np

from dipy.segment.utils import remove_holes_and_islands
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator()
def test_remove_holes_and_islands(rng=None):
    # inefficient but more readable
    temp = rng.choice([0, 1], size=(40, 40, 40), p=[0.8, 0.2])
    temp[6:34, 6:34, 6:34] = 0
    temp[7:33, 7:33, 7:33] = 1
    temp[8:32, 8:32, 8:32] = rng.choice([0, 1], size=(24, 24, 24), p=[0.2, 0.8])
    output = remove_holes_and_islands(temp)
    ground_truth = np.zeros((40, 40, 40))
    ground_truth[7:33, 7:33, 7:33] = 1
    np.testing.assert_equal(output, ground_truth)


def test_remove_holes_and_islands_warnings():
    # Not binary test
    non_binary_img = np.concatenate(
        [np.zeros((30, 30, 10)), np.ones((30, 30, 10)), np.ones((30, 30, 10)) * 2],
        axis=-1,
    )
    np.testing.assert_warns(UserWarning, remove_holes_and_islands, non_binary_img)

    # No background test
    no_background_img = np.ones((40, 40, 40))
    np.testing.assert_warns(UserWarning, remove_holes_and_islands, no_background_img)

    # No foreground test
    no_foreground_img = np.zeros((40, 40, 40))
    np.testing.assert_warns(UserWarning, remove_holes_and_islands, no_foreground_img)
