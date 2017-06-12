import time

import numpy as np
import numpy.testing as npt

import dipy
from dipy.parallel.main import ParallelFunction

class MaxCumSum(ParallelFunction):

    def _main(self, data, weights=1., return_cumsum=False):
        if data.ndim != 1:
            raise ValueError("data must be 1d")
        tmp = data * weights
        cumsum = tmp.cumsum()
        argmax = data.argmax()
        max = data[argmax]
        return {"argmax":argmax, "cumsum":cumsum, "max":max,
                "flag":data[0] == 0}

    def _default_values(self, data, mask, weights, return_cumsum=False):
        res = {"argmax":np.array(-1, int),
               "max":np.array(-999., float)}
        if return_cumsum:
            res["cumsum"] = np.zeros(data.shape[-1], float)
        return res

maxcumsum = MaxCumSum()


def testParallelFunction():

    data = np.random.random((3, 4, 5)) - .5
    mask = np.random.random((3, 4)) > .25
    mask[0, 0] = True
    data[0, 0, 0] = 0.
    weights = np.random.random(5)
    a = maxcumsum(data, mask, weights, return_cumsum=True)
    dipy.parallel.config.activate_multithreading()
    b = maxcumsum(data, mask, weights, return_cumsum=True)

    cumsum = (data * mask[..., None] * weights).cumsum(-1)
    max = data.max(-1)
    argmax = data.argmax(-1)
    max[~mask] = -999
    argmax[~mask] = -1

    npt.assert_array_almost_equal(a["cumsum"], cumsum)
    npt.assert_array_almost_equal(a["max"], max)
    npt.assert_array_almost_equal(a["argmax"], argmax)

    for key in a.keys():
        npt.assert_array_equal(a[key], b[key])

    b = maxcumsum(data, mask, weights, return_cumsum=False)
    assert("cumsum" not in b)


class BrokenFunction(ParallelFunction):

    def _default_values(self, data, mask):
        return {"out":np.zeros(data.shape[-1], float)}

    def _main(self, data):
        raise ValueError

brokenfunction = BrokenFunction()

def test_error_in_function():
    data = np.random.random((3, 4, 5)) - .5
    mask = np.random.random((3, 4)) > .25
    npt.assert_raises(ValueError, brokenfunction, data, mask)
    dipy.parallel.config.deactivate_multithreading()
    npt.assert_raises(ValueError, brokenfunction, data, mask)


if __name__ == "__main__":
    npt.run_module_suite
