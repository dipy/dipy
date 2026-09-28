import numpy as np
import pytest


def test_histeq_reexport_warns():
    from dipy.core.histeq import histeq

    with pytest.warns(DeprecationWarning, match="dipy.core.histeq is deprecated"):
        histeq(np.arange(9.0).reshape(3, 3))
