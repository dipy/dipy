import numpy as np
from dipy.data import get_data
from dipy.core.gradients import gradient_table
from numpy.testing import (assert_almost_equal,
                           assert_equal,
                           run_module_suite)
from dipy.reconst.shore import ShoreModel, SHOREmatrix
from dipy.sims.voxel import (MultiTensor, all_tensor_evecs, multi_tensor_odf, single_tensor_odf,
                            multi_tensor_rtop, multi_tensor_msd, multi_tensor_pdf)
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi
from dipy.data import get_sphere


def test_shore_metrics():
    fetch_taiwan_ntu_dsi()
    img, gtab = read_taiwan_ntu_dsi()

    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))
    angl = [(0, 0), (60, 0)]
    S, sticks = MultiTensor(gtab, mevals, S0=100, angles=angl,
                            fractions=[50, 50], snr=None)
    S = S / S[0, None].astype(np.float)

    radial_order = 8
    zeta = 800
    lambdaN = 1e-12
    lambdaL = 1e-12
    asm = ShoreModel(gtab, radial_order=radial_order, zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    asmfit = asm.fit(S)
    c_shore= asmfit.shore_coeff

    cmat = SHOREmatrix(radial_order, zeta, gtab)
    S_reconst = np.dot(cmat, c_shore)
    nmse_signal = np.sqrt(np.sum((S - S_reconst) ** 2)) / (S.sum())
    assert_almost_equal(nmse_signal, 0.0, 4)

    mevecs2 = np.zeros((2, 3, 3))
    angl = np.array(angl)
    for i in range(2):
        mevecs2[i] = all_tensor_evecs(sticks[i]).T

    sphere = get_sphere('symmetric724')
    v = sphere.vertices
    radius = 10e-3
    pdf_shore = asmfit.pdf(v * radius)
    pdf_mt = multi_tensor_pdf(v * radius, [.5, .5], mevals=mevals, mevecs=mevecs2)
    nmse_pdf = np.sqrt(np.sum((pdf_mt - pdf_shore) ** 2)) / (pdf_mt.sum())
    assert_almost_equal(nmse_pdf, 0.0, 2)

    rtop_shore_signal = asmfit.rtop_signal()
    rtop_shore_pdf = asmfit.rtop_pdf()
    assert_almost_equal(rtop_shore_signal, rtop_shore_pdf, 9)
    rtop_mt = multi_tensor_rtop([.5, .5], mevals=mevals)
    assert_equal(rtop_mt/rtop_shore_signal < 1.12 and rtop_mt/rtop_shore_signal > 0.9 , True)
    
    msd_mt = multi_tensor_msd([.5, .5], mevals=mevals)
    msd_shore = asmfit.msd()
    assert_equal(msd_mt/msd_shore < 1.05 and msd_mt/msd_shore > 0.95 , True)

if __name__ == '__main__':
    run_module_suite()
