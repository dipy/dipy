from os.path import join as pjoin

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

import numpy as np

from nose.tools import eq_
import numpy.testing as npt
from dipy.reconst import mapmri

from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import generate_bvecs
from dipy.workflows.reconst import ReconstMAPMRIFlow


def test_reconst_mmri_laplacian():
    reconst_mmri_core(ReconstMAPMRIFlow, lap=True, pos=False)


def test_reconst_mmri_none():
    reconst_mmri_core(ReconstMAPMRIFlow, lap=False, pos=False)


@np.testing.dec.skipif(not mapmri.have_cvxpy)
def test_reconst_mmri_both():
    reconst_mmri_core(ReconstMAPMRIFlow, lap=True, pos=True)


@np.testing.dec.skipif(not mapmri.have_cvxpy)
def test_reconst_mmri_positivity():
    reconst_mmri_core(ReconstMAPMRIFlow, lap=True, pos=False)


def reconst_mmri_core(flow, lap, pos):
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames('small_25')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()

        mmri_flow = flow()
        mmri_flow.run(data_files=data_path, bvals_files=bval_path,
                      bvecs_files=bvec_path, small_delta=0.0129,
                      big_delta=0.0218, laplacian=lap,
                      positivity=pos, out_dir=out_dir)

        rtop = mmri_flow.last_generated_outputs['out_rtop']
        rtop_data = nib.load(rtop).get_data()
        eq_(rtop_data.shape, volume.shape[:-1])

        lapnorm = mmri_flow.last_generated_outputs['out_lapnorm']
        lapnorm_data = nib.load(lapnorm).get_data()
        eq_(lapnorm_data.shape, volume.shape[:-1])

        msd = mmri_flow.last_generated_outputs['out_msd']
        msd_data = nib.load(msd).get_data()
        eq_(msd_data.shape, volume.shape[:-1])

        qiv = mmri_flow.last_generated_outputs['out_qiv']
        qiv_data = nib.load(qiv).get_data()
        eq_(qiv_data.shape, volume.shape[:-1])

        rtap = mmri_flow.last_generated_outputs['out_rtap']
        rtap_data = nib.load(rtap).get_data()
        eq_(rtap_data.shape, volume.shape[:-1])

        rtpp = mmri_flow.last_generated_outputs['out_rtpp']
        rtpp_data = nib.load(rtpp).get_data()
        eq_(rtpp_data.shape, volume.shape[:-1])

        ng = mmri_flow.last_generated_outputs['out_ng']
        ng_data = nib.load(ng).get_data()
        eq_(ng_data.shape, volume.shape[:-1])

        parng = mmri_flow.last_generated_outputs['out_parng']
        parng_data = nib.load(parng).get_data()
        eq_(parng_data.shape, volume.shape[:-1])

        perng = mmri_flow.last_generated_outputs['out_perng']
        perng_data = nib.load(perng).get_data()
        eq_(perng_data.shape, volume.shape[:-1])

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


if __name__ == '__main__':
    test_reconst_mmri_laplacian()
    test_reconst_mmri_none()
    test_reconst_mmri_positivity()
    test_reconst_mmri_both()
