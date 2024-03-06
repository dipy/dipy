import pytest

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.nn.histo_resdnn import HistoResDNN
from dipy.utils.optpkg import optional_package
import numpy as np
from numpy.testing import assert_almost_equal, assert_raises, assert_equal

tf, have_tf, _ = optional_package('tensorflow', min_version='2.0.0')


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_default_weights():
    input_arr = np.expand_dims(
        np.array([1.15428471, 0.37460899, -0.16743798, -0.02638639,
                  -0.02587842, -0.24743459, -0.11091634, -0.01974129,
                  -0.03463564, 0.04234652, 0.00909119, -0.02181194,
                  -0.01141419, 0.06747056, -0.02881568, 0.0037776,
                  0.02069041, 0.01655271, -0.00958642, 0.0103591,
                  -0.00579612, 0.00559265, 0.00311974, 0.0067629,
                  0.00140297, 0.01844978, -0.00551951, 0.02215372,
                  0.00186543, -0.01057652, 0.00189625, -0.01114438,
                  0.00509697, -0.00150783, 0.01585437, 0.00256389,
                  0.00196107, -0.0108544, 0.01143742, -0.00547229,
                  -0.01040528, 0.0114365, -0.02261801, 0.00452243,
                  0.0015014]), axis=0)

    target_arr = np.expand_dims(
        np.array([0.17804961, -0.18878266, 0.02026339, -0.0100488,
                  -0.04045521, 0.20264171, -0.16151273, 0.00508221,
                  0.01158303, 0.03848331, 0.04242867, -0.00493216,
                  -0.05138939, 0.03944791, -0.06210141, -0.01777741,
                  0.00032369, -0.00781484, 0.02685455, 0.00617174,
                  -0.01357785, 0.0112316, 0.02457713, -0.00307974,
                  -0.00110319, -0.0274653, 0.01606723, -0.05088685,
                  0.0017358, -0.00533427, -0.00785866, 0.00529946,
                  0.00624491, 0.00682212, -0.00551173, -0.00760572,
                  -0.00145562, 0.02271283, 0.0238023, -0.01574752,
                  0.00853913, -0.00715324, 0.02677651, 0.01718479,
                  -0.01433261]), axis=0)

    resdnn_model = HistoResDNN()
    resdnn_model.fetch_default_weights()
    results_arr = resdnn_model._HistoResDNN__predict(input_arr)
    assert_almost_equal(results_arr, target_arr)


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_predict_shape_and_masking():
    resdnn_model = HistoResDNN()
    resdnn_model.fetch_default_weights()

    dwi_fname, bval_fname, bvec_fname = get_fnames('stanford_hardi')
    data, _ = load_nifti(dwi_fname)
    data = np.squeeze(data)
    bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    mask = np.zeros(data.shape[0:3], dtype=bool)
    mask[38:40, 45:50, 35:40] = 1

    results_arr = resdnn_model.predict(data, gtab, mask=mask)
    results_pos = np.sum(results_arr, axis=-1, dtype=bool)
    assert_equal(mask, results_pos)
    assert_equal(results_arr.shape[-1], 45)


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_wrong_sh_order_weights():
    resdnn_model = HistoResDNN(sh_order_max=6)
    fetch_model_weights_path = get_fnames('histo_resdnn_weights')
    assert_raises(ValueError, resdnn_model.load_model_weights,
                  fetch_model_weights_path)


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_wrong_sh_order_input():
    resdnn_model = HistoResDNN()
    fetch_model_weights_path = get_fnames('histo_resdnn_weights')
    resdnn_model.load_model_weights(fetch_model_weights_path)
    assert_raises(ValueError, resdnn_model._HistoResDNN__predict,
                  np.zeros((1, 28)))
