"""File dedicated to test ``dipy.core.rng`` module."""

import numpy as np
import numpy.testing as npt

from dipy.core import rng as core_rng
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator()
def test_wichmann_hill2006(rng):
    # These functions are stateless: same seeds → same value each call.
    # Use varied seeds to generate a sequence and verify outputs are in [0, 1].
    seeds = rng.integers(1, 2**20, size=(100, 4))
    n_generated = np.array(
        [
            core_rng.WichmannHill2006(
                ix=int(s[0]), iy=int(s[1]), iz=int(s[2]), it=int(s[3])
            )
            for s in seeds
        ]
    )
    assert np.all((n_generated >= 0) & (n_generated <= 1))
    npt.assert_raises(ValueError, core_rng.WichmannHill2006, ix=0)


@set_random_number_generator()
def test_wichmann_hill1982(rng):
    seeds = rng.integers(1, 2**15, size=(100, 3))
    n_generated = np.array(
        [
            core_rng.WichmannHill1982(ix=int(s[0]), iy=int(s[1]), iz=int(s[2]))
            for s in seeds
        ]
    )
    assert np.all((n_generated >= 0) & (n_generated <= 1))
    npt.assert_raises(ValueError, core_rng.WichmannHill1982, iz=0)


@set_random_number_generator()
def test_LEcuyer(rng):
    seeds = rng.integers(1, 2**20, size=(100, 2))
    n_generated = np.array(
        [core_rng.LEcuyer(s1=int(s[0]), s2=int(s[1])) for s in seeds]
    )
    assert np.all((n_generated >= 0) & (n_generated <= 1))
    npt.assert_raises(ValueError, core_rng.LEcuyer, s2=0)
