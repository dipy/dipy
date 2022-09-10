import tempfile
import os
from numpy.testing import assert_raises, assert_equal, run_module_suite
import pytest

from dipy.utils.optpkg import optional_package
from dipy.data import read_five_af_bundles

fury, has_fury, _ = optional_package('fury')
if has_fury:
    from dipy.viz.streamline import show_bundles


bundles = read_five_af_bundles()


@pytest.mark.skipif(not has_fury, reason='Requires FURY')
def test_output_created():
    views = ['axial', 'sagital', 'coronal']

    colors = [[0.91, 0.26, 0.35], [0.99, 0.50, 0.38], [0.99, 0.88, 0.57],
              [0.69, 0.85, 0.64], [0.51, 0.51, 0.63]]

    with tempfile.TemporaryDirectory() as temp_dir:
        for view in views:
            fname = os.path.join(temp_dir, f'test_{view}.png')
            show_bundles(bundles, False, view=view, fname=fname)
            assert_equal(True, os.path.exists(fname))

        fname = os.path.join(temp_dir, 'test_colors.png')
        show_bundles(bundles, False, colors=colors, fname=fname)
        assert_equal(True, os.path.exists(fname))


@pytest.mark.skipif(not has_fury, reason='Requires FURY')
def test_incorrect_view():
    assert_raises(ValueError, show_bundles, bundles, False, 'wrong_view')


if __name__ == '__main__':
    run_module_suite()
