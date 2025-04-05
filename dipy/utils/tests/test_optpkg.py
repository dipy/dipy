import os
from tempfile import TemporaryDirectory

import pytest

from dipy.testing import assert_false, assert_true
from dipy.utils.optpkg import optional_package
from dipy.utils.tripwire import TripWireError


def test_optional_package():
    pkg, have_pkg, setup_module = optional_package("os")
    assert_true(have_pkg)
    assert_true(hasattr(pkg, "path"))

    pkg, have_pkg, setup_module = optional_package("not_a_package")
    assert_false(have_pkg)
    with pytest.raises(TripWireError):
        pkg.some_function()

    pkg, have_pkg, setup_module = optional_package("dipy", min_version="10.0.0")
    assert_false(have_pkg)
    with pytest.raises(TripWireError):
        pkg.some_function()

    pkg, have_pkg, setup_module = optional_package("dipy", min_version="1.0.0")
    assert_true(have_pkg)

    with TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "fake_package"))
        open(os.path.join(tmpdir, "fake_package", "__init__.py"), "a").close()
        current_dir = os.getcwd()
        os.chdir(tmpdir)
        pkg, have_pkg, setup_module = optional_package(
            "fake_package", min_version="1.0.0"
        )
        assert_false(have_pkg)
        assert_false(hasattr(pkg, "__version__"))
        with pytest.raises(TripWireError):
            pkg.some_function()
        os.chdir(current_dir)
