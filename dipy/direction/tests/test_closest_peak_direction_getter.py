import warnings

import numpy.testing as npt

from dipy.direction.closest_peak_direction_getter import BaseDirectionGetter
from dipy.direction.closest_peak_direction_getter import BasePmfDirectionGetter


def test_BaseDirectionGetter_deprecated_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # create a BasePmfDirectionGetter from deprecated class name
        dg = BaseDirectionGetter(None, 0, None)
        npt.assert_equal(len(w), 1)
        npt.assert_equal(issubclass(w[-1].category, DeprecationWarning), True)
        npt.assert_equal("deprecated" in str(w[-1].message), True)

    npt.assert_equal(isinstance(dg, BasePmfDirectionGetter), True)
