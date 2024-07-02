import warnings

import numpy as np
import numpy.testing as npt

from dipy.core.sphere import unit_octahedron
from dipy.core.sphere import HemiSphere
from dipy.data import get_fnames, get_sphere
from dipy.direction.pmf import SHCoeffPmfGen, SimplePmfGen
from dipy.reconst.shm import (
    SphHarmFit,
    SphHarmModel,
    descoteaux07_legacy_msg,
)
from dipy.tracking.tracker_probabilistic cimport probabilistic_tracker

from dipy.tracking.tracker_parameters import generate_tracking_parameters

from dipy.tracking.tests.test_fast_tracking import get_fast_tracking_performances


def test_ptt_performances():
    # Test ptt tracker on the DiSCo dataset

    params = generate_tracking_parameters("ptt",
                                          max_len=500,
                                          step_size=0.2,
                                          voxel_size=np.ones(3),
                                          max_angle=15,
                                          probe_quality=4)

    r = get_fast_tracking_performances(params)
    npt.assert_(r > 0.85, msg="PTT tracker has a low performance "
                              "score: " + str(r))
