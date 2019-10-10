
import warnings
import pytest

from dipy.utils.optpkg import optional_package
fury, has_fury, _ = optional_package('fury')


@pytest.mark.skipif(has_fury, reason="Skipped because Fury is installed")
def test_viz_import_warning():
    with warnings.catch_warnings(record=True) as w:
        import dipy.viz
        assert len(w) == 1
