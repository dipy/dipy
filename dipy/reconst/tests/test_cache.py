from dipy.reconst.cache import Cache
from dipy.core.sphere import Sphere

from numpy.testing import assert_, assert_equal, run_module_suite


class TestModel(Cache):
    def __init__(self):
        pass


def test_basic_cache():
    t = TestModel()
    s = Sphere(theta=[0], phi=[0])

    assert_(t.cache_get("design_matrix", s) is None)

    m = [[1, 0], [0, 1]]

    t.cache_set("design_matrix", key=s, value=m)
    assert_equal(t.cache_get("design_matrix", s), m)

    t.cache_clear()
    assert_(t.cache_get("design_matrix", s) is None)


if __name__ == "__main__":
    run_module_suite()
