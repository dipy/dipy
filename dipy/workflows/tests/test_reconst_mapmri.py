from os.path import join

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

import numpy as np

from dipy.utils.testing import assert_equal
from dipy.reconst import mapmri

from dipy.data import get_data
from dipy.workflows.reconst import ReconstMAPMRIFlow

import pytest


def test_reconst_mmri_laplacian():
    reconst_mmri_core(ReconstMAPMRIFlow, lap=True, pos=False)


def test_reconst_mmri_none():
    reconst_mmri_core(ReconstMAPMRIFlow, lap=False, pos=False)


@pytest.mark.skipif(not mapmri.have_cvxpy, reason="Requires cvxpy")
def test_reconst_mmri_both():
    reconst_mmri_core(ReconstMAPMRIFlow, lap=True, pos=True)


@pytest.mark.skipif(not mapmri.have_cvxpy, reason="Requires cvxpy")
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
        assert_equal(rtop_data.shape, volume.shape[:-1])

        lapnorm = mmri_flow.last_generated_outputs['out_lapnorm']
        lapnorm_data = nib.load(lapnorm).get_data()
        assert_equal(lapnorm_data.shape, volume.shape[:-1])

        msd = mmri_flow.last_generated_outputs['out_msd']
        msd_data = nib.load(msd).get_data()
        assert_equal(msd_data.shape, volume.shape[:-1])

        qiv = mmri_flow.last_generated_outputs['out_qiv']
        qiv_data = nib.load(qiv).get_data()
        assert_equal(qiv_data.shape, volume.shape[:-1])

        rtap = mmri_flow.last_generated_outputs['out_rtap']
        rtap_data = nib.load(rtap).get_data()
        assert_equal(rtap_data.shape, volume.shape[:-1])

        rtpp = mmri_flow.last_generated_outputs['out_rtpp']
        rtpp_data = nib.load(rtpp).get_data()
        assert_equal(rtpp_data.shape, volume.shape[:-1])

        ng = mmri_flow.last_generated_outputs['out_ng']
        ng_data = nib.load(ng).get_data()
        assert_equal(ng_data.shape, volume.shape[:-1])

        parng = mmri_flow.last_generated_outputs['out_parng']
        parng_data = nib.load(parng).get_data()
        assert_equal(parng_data.shape, volume.shape[:-1])

        perng = mmri_flow.last_generated_outputs['out_perng']
        perng_data = nib.load(perng).get_data()
        assert_equal(perng_data.shape, volume.shape[:-1])


if __name__ == '__main__':
    test_reconst_mmri_laplacian()
    test_reconst_mmri_none()
    test_reconst_mmri_positivity()
    test_reconst_mmri_both()
