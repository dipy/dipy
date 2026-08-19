import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dipy.core.gradients import generate_bvecs
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data
from dipy.reconst import mapmri
from dipy.testing import assert_warns
from dipy.workflows.reconst import ReconstMAPMRIFlow


def test_reconst_mmri_laplacian(tmp_path):
    reconst_mmri_core(tmp_path, ReconstMAPMRIFlow, lap=True, pos=False)


def test_reconst_mmri_none(tmp_path):
    reconst_mmri_core(tmp_path, ReconstMAPMRIFlow, lap=False, pos=False)


@pytest.mark.skipif(not mapmri.have_cvxpy, reason="Requires CVXPY")
def test_reconst_mmri_both(tmp_path):
    reconst_mmri_core(tmp_path, ReconstMAPMRIFlow, lap=True, pos=True)


@pytest.mark.skipif(not mapmri.have_cvxpy, reason="Requires CVXPY")
def test_reconst_mmri_positivity(tmp_path):
    reconst_mmri_core(tmp_path, ReconstMAPMRIFlow, lap=True, pos=False)


def reconst_mmri_core(tmp_path, flow, lap, pos):
    data_path, bval_path, bvec_path = get_fnames(name="small_25")
    volume = load_nifti_data(data_path)

    mmri_flow = flow()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Optimization did not find a solution",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Solution may be inaccurate.*",
            category=UserWarning,
        )
        mmri_flow.run(
            data_files=data_path,
            bvals_files=bval_path,
            bvecs_files=bvec_path,
            small_delta=0.0129,
            big_delta=0.0218,
            laplacian=lap,
            positivity=pos,
            out_dir=tmp_path,
        )

    for out_name in [
        "out_rtop",
        "out_lapnorm",
        "out_msd",
        "out_qiv",
        "out_rtap",
        "out_rtpp",
        "out_ng",
        "out_parng",
        "out_perng",
    ]:
        out_path = mmri_flow.last_generated_outputs[out_name]
        out_data = load_nifti_data(out_path)
        npt.assert_equal(out_data.shape, volume.shape[:-1])

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    bvals[0] = 5.0
    bvecs = generate_bvecs(len(bvals))
    tmp_bval_path = tmp_path / "tmp.bval"
    tmp_bvec_path = tmp_path / "tmp.bvec"
    np.savetxt(tmp_bval_path, bvals)
    np.savetxt(tmp_bvec_path, bvecs.T)
    mmri_flow._force_overwrite = True
    with npt.assert_raises(BaseException):
        assert_warns(
            UserWarning,
            mmri_flow.run,
            data_path,
            tmp_bval_path,
            tmp_bvec_path,
            small_delta=0.0129,
            big_delta=0.0218,
            laplacian=lap,
            positivity=pos,
            out_dir=tmp_path,
        )
