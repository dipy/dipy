from os.path import join

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

import numpy as np

from nose.tools import eq_
from dipy.reconst import mapmri

from dipy.data import get_data
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
        data_path, bval_path, bvec_path = get_data('small_25')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()

        mmri_flow = flow()
        mmri_flow.run(data_file=data_path, data_bvals=bval_path,
                      data_bvecs=bvec_path, small_delta=0.0129,
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


if __name__ == '__main__':
    test_reconst_mmri_laplacian()
    test_reconst_mmri_none()
    test_reconst_mmri_positivity()
    test_reconst_mmri_both()
