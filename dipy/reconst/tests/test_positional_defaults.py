"""Legacy positional spellings of reconst model options are still reported.

Several models used to declare their options before ``*args``, e.g. DIPY 1.11's
``TensorModel(gtab, fit_method="WLS", return_S0_hat=False, *args, **kwargs)``.
Since 1.12.0 those options sit after ``*args``, so ``TensorModel(gtab, "OLS")``
no longer selects a fit method -- it lands in ``*args`` and the model quietly
keeps the default. ``*args`` makes the two spellings indistinguishable, so
``warning_for_keywords`` cannot rewrite the call; it reports it instead, and
these tests pin that report to every model that made the move.
"""

import warnings

import numpy as np
import pytest

from dipy.core.gradients import gradient_table
from dipy.reconst.cti import CorrelationTensorModel
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.reconst.dki_micro import KurtosisMicrostructureModel
from dipy.reconst.dti import TensorModel, design_matrix, wls_fit_tensor
from dipy.reconst.fwdti import FreeWaterTensorModel
from dipy.reconst.msdki import MeanDiffusionKurtosisModel


def _gtab():
    """Return a minimal two-shell gradient table."""
    bvals = np.array([0, 1000, 1000, 2000, 2000] * 3, dtype=float)
    bvecs = np.tile(np.eye(3), (5, 1))
    return gradient_table(bvals=bvals, bvecs=bvecs)


def _user_warnings(record):
    """Return the ``UserWarning`` messages of a ``catch_warnings`` record."""
    return [str(w.message) for w in record if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize(
    ("model", "n_gtab", "legacy_value", "shadowed"),
    [
        (DiffusionKurtosisModel, 1, "OLS", "fit_method"),
        (KurtosisMicrostructureModel, 1, "OLS", "fit_method"),
        (TensorModel, 1, "OLS", "fit_method"),
        (FreeWaterTensorModel, 1, "WLS", "fit_method"),
        (CorrelationTensorModel, 2, "OLS", "fit_method"),
        (MeanDiffusionKurtosisModel, 1, 1.0, "bmag"),
    ],
)
def test_legacy_positional_option_is_reported(model, n_gtab, legacy_value, shadowed):
    gtab = _gtab()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model(*([gtab] * n_gtab), legacy_value)
    messages = _user_warnings(record)
    assert len(messages) == 1, messages
    assert "absorbed by '*args'" in messages[0]
    assert shadowed in messages[0]


@pytest.mark.parametrize(
    ("model", "n_gtab", "option"),
    [
        (DiffusionKurtosisModel, 1, {"fit_method": "OLS"}),
        (KurtosisMicrostructureModel, 1, {"fit_method": "OLS"}),
        (TensorModel, 1, {"fit_method": "OLS"}),
        (FreeWaterTensorModel, 1, {"fit_method": "WLS"}),
        (CorrelationTensorModel, 2, {"fit_method": "OLS"}),
        (MeanDiffusionKurtosisModel, 1, {"bmag": 1.0}),
    ],
)
def test_keyword_spelling_is_silent(model, n_gtab, option):
    gtab = _gtab()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model(*([gtab] * n_gtab), **option)
    assert _user_warnings(record) == []


def test_iter_fit_tensor_reports_legacy_return_s0_hat():
    """``iter_fit_tensor`` wraps the fit methods, so it needs the report too."""
    gtab = _gtab()
    dmat = design_matrix(gtab)
    rng = np.random.default_rng(1234)
    data = np.abs(rng.normal(100, 5, size=(4, len(gtab.bvals))))

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        wls_fit_tensor(dmat, data, return_S0_hat=True)
    assert _user_warnings(record) == []

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        wls_fit_tensor(dmat, data, True)
    messages = _user_warnings(record)
    assert any(
        "absorbed by '*args'" in msg and "return_S0_hat" in msg for msg in messages
    ), messages
