import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from dipy.align.aniso2iso import resample
from dipy.viz import fvtk
from dipy.core.ndindex import ndindex
from dipy.data import get_sphere
import dipy.reconst.dti as dti
from dipy.data import get_data, dsi_voxels
from dipy.reconst.dsi import (
    DiffusionSpectrumModel, DiffusionSpectrumDeconvModel)
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           run_module_suite,
                           assert_array_equal,
                           assert_raises)
from dipy.data import get_data, dsi_voxels
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.odf import gfa, peak_directions
from dipy.sims.voxel import SticksAndBall
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from numpy.testing import assert_equal
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.core.sphere_stats import angular_similarity


def test_dsi():
    # load symmetric 724 sphere
    sphere = get_sphere('symmetric724')
    # load icosahedron sphere
    sphere2 = create_unit_sphere(5)
    btable = np.loadtxt(get_data('dsi515btable'))
    gtab = gradient_table(btable[:, 0], btable[:, 1:])
    data, golden_directions = SticksAndBall(gtab, d=0.0015,
                                            S0=100, angles=[(0, 0), (90, 0)],
                                            fractions=[50, 50], snr=None)
    data = data / float(data[0])
    dsmodel = DiffusionSpectrumModel(gtab)
    rtop_signal_norm = dsmodel.fit(data).rtop_signal()
    rtop_pdf_norm = dsmodel.fit(data).rtop_pdf()
    rtop_pdf = dsmodel.fit(data).rtop_pdf(normalized=False)
    MSD_norm = dsmodel.fit(data).msd_discrete()
    MSD = dsmodel.fit(data).msd_discrete(normalized=False)
    assert_almost_equal(rtop_signal_norm, rtop_pdf, 10)

    # We need a test for MSD!!!


if __name__ == '__main__':
    run_module_suite()
