import numpy as np
import pytest

from dipy.core.histeq import histeq


def test_histeq_reexport_warns():
    with pytest.warns(DeprecationWarning, match="dipy.core.histeq is deprecated"):
        histeq(np.arange(9.0).reshape(3, 3))
