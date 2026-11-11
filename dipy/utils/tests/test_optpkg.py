import re
import sys

import pytest

from dipy import __version__ as dipy_version
from dipy.testing import assert_false, assert_true
from dipy.utils.optpkg import optional_package
from dipy.utils.tripwire import TripWireError


def assert_tripwire_message(pkg, expected_msg):
    """Verify TripWireError message for a proxied package."""
    with pytest.raises(TripWireError, match=re.escape(expected_msg)):
        pkg.some_function()


def test_optional_package(tmp_path):
    pkg, have_pkg, _ = optional_package("os")
    assert_true(have_pkg)
    assert_true(hasattr(pkg, "path"))

    pkg, have_pkg, _ = optional_package("not_a_package")
    assert_false(have_pkg)
    assert_tripwire_message(
        pkg,
        "We need package not_a_package for these functions, but "
        "``import not_a_package`` raised an ImportError",
    )

    pkg, have_pkg, _ = optional_package("dipy", min_version="10.0.0")
    assert_false(have_pkg)
    assert_tripwire_message(
        pkg,
        "We need package dipy version >= 10.0.0 for these functions, but you have "
        f"version {dipy_version} installed.",
    )

    pkg, have_pkg, _ = optional_package("dipy", max_version="0.1.0")
    assert_false(have_pkg)
    assert_tripwire_message(
        pkg,
        "We need package dipy version <= 0.1.0 for these functions, but you have "
        f"version {dipy_version} installed.",
    )

    pkg, have_pkg, _ = optional_package(
        "dipy", min_version="0.0.1", max_version="0.0.2"
    )
    assert_false(have_pkg)
    assert_tripwire_message(
        pkg,
        "We need package dipy version >= 0.0.1 and <= 0.0.2 for these functions, "
        f"but you have version {dipy_version} installed.",
    )

    pkg, have_pkg, _ = optional_package("dipy", min_version="1.0.0")
    assert_true(have_pkg)

    pkg, have_pkg, _ = optional_package(
        "dipy", min_version="1.0.0", max_version="100.0.0"
    )
    assert_true(have_pkg)

    pkg, have_pkg, _ = optional_package(
        "dipy", min_version="100.0.0", max_version="1.0.0"
    )
    assert_false(have_pkg)
    assert_tripwire_message(
        pkg,
        "Invalid version requirements for package dipy: "
        "min_version 100.0.0 > max_version 1.0.0",
    )

    fake_pkg = tmp_path / "fake_package"
    fake_pkg.mkdir()
    init_file = fake_pkg / "__init__.py"
    init_file.touch()

    old_sys_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))

    pkg, have_pkg, _ = optional_package("fake_package", min_version="1.0.0")
    assert_false(have_pkg)
    assert_false(hasattr(pkg, "__version__"))
    msg = (
        "We need package fake_package version >= 1.0.0 for these functions, "
        "but you have version 0.0.0 installed. Your installation might be "
        "incomplete or corrupted."
    )
    assert_tripwire_message(pkg, msg)

    sys.modules.pop("fake_package", None)
    sys.path = old_sys_path
