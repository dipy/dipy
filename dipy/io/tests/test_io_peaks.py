import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, run_module_suite,
                           assert_equal, assert_)
from dipy.reconst.odf import (OdfFit, OdfModel, gfa)

from dipy.direction.peaks import (peaks_from_model,
                                  peak_directions,
                                  peak_directions_nl,
                                  reshape_peaks_for_visualization)
from dipy.core.subdivide_octahedron import create_unit_hemisphere
from dipy.core.sphere import unit_icosahedron
from dipy.sims.voxel import multi_tensor, all_tensor_evecs, multi_tensor_odf
from dipy.data import get_data, get_sphere
from dipy.core.gradients import gradient_table, GradientTable
from dipy.core.sphere_stats import angular_similarity
from ipdb import set_trace


def test_peaks_save_load():

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

    from dipy.reconst.shm import CsaOdfModel

    model = CsaOdfModel(gtab, 4)

    pam = peaks_from_model(model, data[None, :], sphere, .5, 45,
                           return_odf=True, return_sh=True)

    np.savez_compressed('peaks.npz',
                        affine=pam.affine,
                        peak_dirs=pam.peak_dirs,
                        peak_values=pam.peak_values,
                        peak_indices=pam.peak_indices,
                        shm_coeff=pam.shm_coeff,
                        sphere=pam.sphere,
                        B=pam.B,
                        total_weight=pam.total_weight,
                        ang_thr=pam.ang_thr,
                        gfa=pam.gfa,
                        qa=pam.qa)

    #print(outfile)

    set_trace()
    # Test that spherical harmonic coefficients return back correctly
    odf2 = np.dot(pam.shm_coeff, pam.B)
    assert_array_almost_equal(pam.odf, odf2)
    assert_equal(pam.shm_coeff.shape[-1], 45)

    pam = peaks_from_model(model, data[None, :], sphere, .5, 45,
                           return_odf=True, return_sh=False)
    assert_equal(pam.shm_coeff, None)

    pam = peaks_from_model(model, data[None, :], sphere, .5, 45,
                           return_odf=True, return_sh=True,
                           sh_basis_type='mrtrix')

    odf2 = np.dot(pam.shm_coeff, pam.B)
    assert_array_almost_equal(pam.odf, odf2)



test_peaks_save_load()
