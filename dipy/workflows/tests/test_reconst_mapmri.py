from os.path import join as pjoin
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data
from dipy.core.gradients import generate_bvecs
from dipy.reconst import mapmri
from dipy.workflows.reconst import ReconstMAPMRIFlow


def test_reconst_mmri_laplacian():
    reconst_mmri_core(ReconstMAPMRIFlow, lap=True, pos=False)


def test_reconst_mmri_none():
    reconst_mmri_core(ReconstMAPMRIFlow, lap=False, pos=False)


@pytest.mark.skipif(not mapmri.have_cvxpy, reason='Requires CVXPY')
def test_reconst_mmri_both():
    reconst_mmri_core(ReconstMAPMRIFlow, lap=True, pos=True)


@pytest.mark.skipif(not mapmri.have_cvxpy, reason='Requires CVXPY')
def test_reconst_mmri_positivity():
    reconst_mmri_core(ReconstMAPMRIFlow, lap=True, pos=False)


def reconst_mmri_core(flow, lap, pos):
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames('small_25')
        volume = load_nifti_data(data_path)

        mmri_flow = flow()

        msg = "Optimization did not find a solution"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=msg,
                                    category=UserWarning)
            mmri_flow.run(data_files=data_path, bvals_files=bval_path,
                          bvecs_files=bvec_path, small_delta=0.0129,
                          big_delta=0.0218, laplacian=lap,
                          positivity=pos, out_dir=out_dir)

        for out_name in ['out_rtop', 'out_lapnorm', 'out_msd', 'out_qiv',
                         'out_rtap', 'out_rtpp', 'out_ng', 'out_parng',
                         'out_perng']:
            out_path = mmri_flow.last_generated_outputs[out_name]
            out_data = load_nifti_data(out_path)
            npt.assert_equal(out_data.shape, volume.shape[:-1])

        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        bvals[0] = 5.
        bvecs = generate_bvecs(len(bvals))
        tmp_bval_path = pjoin(out_dir, "tmp.bval")
        tmp_bvec_path = pjoin(out_dir, "tmp.bvec")
        np.savetxt(tmp_bval_path, bvals)
        np.savetxt(tmp_bvec_path, bvecs.T)
        mmri_flow._force_overwrite = True
        with npt.assert_raises(BaseException):
            npt.assert_warns(UserWarning, mmri_flow.run, data_path,
                             tmp_bval_path, tmp_bvec_path, small_delta=0.0129,
                             big_delta=0.0218, laplacian=lap,
                             positivity=pos, out_dir=out_dir)
