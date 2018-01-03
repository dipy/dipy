from os.path import join

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

import numpy as np

from nose.tools import eq_
from dipy.reconst import mapmri

from dipy.data import get_data
from dipy.workflows.reconst import ReconstMAPMRILaplacian, ReconstMAPMRIPositivity, ReconstMAPMRIBoth


def test_reconst_mmri_laplacian():
    reconst_mmri_core(ReconstMAPMRILaplacian)

@np.testing.dec.skipif(not mapmri.have_cvxpy)
def test_reconst_mmri_both():
    reconst_mmri_core(ReconstMAPMRIBoth)

@np.testing.dec.skipif(not mapmri.have_cvxpy)
def test_reconst_mmri_positivity():
    reconst_mmri_core(ReconstMAPMRIPositivity)


def reconst_mmri_core(flow):
    with TemporaryDirectory() as out_dir:
        data_path, bvec_path, bval_path = get_data('small_25')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()

        mmri_flow = flow()

        mmri_flow.run(data_file=data_path, data_bvals=bval_path, data_bvecs=bvec_path, out_dir=out_dir)

        rtop = mmri_flow.last_generated_outputs['out_rtop']
        print(rtop)
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


if __name__ == '__main__':
    test_reconst_mmri_laplacian()
    test_reconst_mmri_positivity()
    test_reconst_mmri_both()