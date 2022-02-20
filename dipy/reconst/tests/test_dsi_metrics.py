import numpy as np
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.data import get_fnames
from dipy.core.gradients import gradient_table
from numpy.testing import assert_almost_equal
from dipy.sims.voxel import sticks_and_ball, multi_tensor


def test_dsi_metrics():
    btable = np.loadtxt(get_fnames('dsi4169btable'))
    gtab = gradient_table(btable[:, 0], btable[:, 1:])
    data, _ = sticks_and_ball(gtab, d=0.0015, S0=100,
                              angles=[(0, 0), (60, 0)],
                              fractions=[50, 50], snr=None)

    dsmodel = DiffusionSpectrumModel(gtab, qgrid_size=21, filter_width=4500)
    rtop_signal_norm = dsmodel.fit(data).rtop_signal()
    dsmodel.fit(data).rtop_pdf()
    rtop_pdf = dsmodel.fit(data).rtop_pdf(normalized=False)
    assert_almost_equal(rtop_signal_norm, rtop_pdf, 6)
    dsmodel = DiffusionSpectrumModel(gtab, qgrid_size=21, filter_width=4500)
    mevals = np.array(([0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]))
    S_0, _ = multi_tensor(gtab, mevals, S0=100,
                          angles=[(0, 0), (60, 0)],
                          fractions=[50, 50], snr=None)
    S_1, _ = multi_tensor(gtab, mevals * 2.0, S0=100,
                          angles=[(0, 0), (60, 0)],
                          fractions=[50, 50], snr=None)
    MSD_norm_0 = dsmodel.fit(S_0).msd_discrete(normalized=True)
    MSD_norm_1 = dsmodel.fit(S_1).msd_discrete(normalized=True)
    assert_almost_equal(MSD_norm_0, 0.5 * MSD_norm_1, 4)
