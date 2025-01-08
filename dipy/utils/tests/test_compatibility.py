from numpy.testing import assert_equal

from dipy.utils.compatibility import check_max_version, check_min_version


def test_check_min_version():
    assert_equal(check_min_version("dipy", "15.0.0"), False)
    assert_equal(check_min_version("dipy", "0.8.0"), True)

    assert_equal(check_min_version("numpy", "15.0.0"), False)
    assert_equal(check_min_version("numpy", "1.8.0"), True)

    assert_equal(check_min_version("scipy", "3.0.0"), False)
    assert_equal(check_min_version("scipy", "1.8.0"), True)

    assert_equal(check_min_version("fakepackage", "3.0.0"), False)
    assert_equal(check_min_version("fakepackage", "0.8.0"), False)


def test_check_max_version():
    assert_equal(check_max_version("dipy", "15.0.0"), True)
    assert_equal(check_max_version("dipy", "0.8.0"), False)

    assert_equal(check_max_version("numpy", "15.0.0"), True)
    assert_equal(check_max_version("numpy", "1.8.0"), False)

    assert_equal(check_max_version("scipy", "3.0.0"), True)
    assert_equal(check_max_version("scipy", "1.8.0"), False)

    assert_equal(check_max_version("fakepackage", "3.0.0"), False)
    assert_equal(check_max_version("fakepackage", "0.8.0"), False)
