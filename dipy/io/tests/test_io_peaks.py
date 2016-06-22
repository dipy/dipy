import os
import numpy as np
import numpy.testing as npt
from dipy.reconst.odf import (OdfFit, OdfModel, gfa)

from dipy.direction.peaks import (peaks_from_model,
                                  peak_directions,
                                  peak_directions_nl,
                                  reshape_peaks_for_visualization)
from dipy.core.subdivide_octahedron import create_unit_hemisphere
from dipy.core.sphere import unit_icosahedron
from dipy.sims.voxel import multi_tensor, all_tensor_evecs, multi_tensor_odf
from dipy.data import get_data, get_sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.core.gradients import gradient_table, GradientTable
from dipy.io.peaks import load_peaks, save_peaks, peaks_to_niftis
from nibabel.tmpdirs import TemporaryDirectory
from ipdb import set_trace


def test_peaks_save_load():

    with TemporaryDirectory() as tmpdir:

        SNR = 100
        S0 = 100

        _, fbvals, fbvecs = get_data('small_64D')

        sphere = get_sphere('repulsion724')

        bvals = np.load(fbvals)
        bvecs = np.load(fbvecs)

        gtab = gradient_table(bvals, bvecs)
        mevals = np.array(([0.0015, 0.0003, 0.0003],
                           [0.0015, 0.0003, 0.0003]))

        data, _ = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (60, 0)],
                               fractions=[50, 50], snr=SNR)

        model = CsaOdfModel(gtab, 4)

        pam = peaks_from_model(model, data[None, :], sphere, .5, 45,
                               return_odf=True, return_sh=True)
        save_peaks(os.path.join(tmpdir, 'peaks.npz'), pam, True)

        pam2 = load_peaks(os.path.join(tmpdir, 'peaks.npz'))
        npt.assert_array_equal(pam.shm_coeff, pam2.shm_coeff)

        odf2 = np.dot(pam.shm_coeff, pam.B)
        npt.assert_array_almost_equal(pam.odf, odf2)
        npt.assert_equal(pam.shm_coeff.shape[-1], 45)

        pam = peaks_from_model(model, data[None, :], sphere, .5, 45,
                               return_odf=True, return_sh=False)
        npt.assert_equal(pam.shm_coeff, None)

        pam = peaks_from_model(model, data[None, :], sphere, .5, 45,
                               return_odf=True, return_sh=True,
                               sh_basis_type='mrtrix')

        odf2 = np.dot(pam.shm_coeff, pam.B)
        npt.assert_array_almost_equal(pam.odf, odf2)


npt.run_module_suite()
