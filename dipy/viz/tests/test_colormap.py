import numpy as np

from nose.tools import assert_equal, assert_false
from numpy.testing import assert_array_equal

from dipy.viz.colormap import distinguishable_colormap


def test_distinguishable_colormap():
    # Test limiting the number of colors
    nb_colors = 10
    colors = distinguishable_colormap(nb_colors=nb_colors)
    assert_equal(len(colors), nb_colors)

    # Colors should be the same whether or not `nb_colors` if provided.
    colormap = distinguishable_colormap()
    for c1, c2 in zip(colormap, colors):
        assert_array_equal(c1, c2)

    # Test excluding the `nb_colors` first colors.
    colors_excluded = np.asarray(colors)
    colors = distinguishable_colormap(exclude=colors, nb_colors=1000)

    # After 1000 colors the L1 distance in the RGB space is still higher than 1e-2.
    for i, c in enumerate(colors):
        matches = np.all(np.abs(colors_excluded - np.asarray([c])) < 1e-2, axis=1)
        assert_false(np.any(matches))
