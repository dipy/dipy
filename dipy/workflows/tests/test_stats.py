#!/usr/bin/env python

import os
from os.path import join
from dipy.utils.optpkg import optional_package
import numpy.testing as npt
import pytest
from nibabel.tmpdirs import TemporaryDirectory
import numpy as np
from dipy.tracking.streamline import Streamlines
from dipy.testing import assert_true
from dipy.io.image import save_nifti, load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.data import get_fnames
from dipy.workflows.stats import SNRinCCFlow
from dipy.workflows.stats import BundleAnalysisPopulationFlow
from dipy.workflows.stats import LinearMixedModelsFlow
pd, have_pandas, _ = optional_package("pandas")
_, have_statsmodels, _ = optional_package("statsmodels")
_, have_tables, _ = optional_package("tables")


def test_stats():
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames('small_101D')
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0], dtype=np.uint8)
        mask_path = join(out_dir, 'tmp_mask.nii.gz')
        save_nifti(mask_path, mask, affine)

        snr_flow = SNRinCCFlow(force=True)
        args = [data_path, bval_path, bvec_path, mask_path]

        snr_flow.run(*args, out_dir=out_dir)
        assert_true(os.path.exists(os.path.join(out_dir, 'product.json')))
        assert_true(os.stat(os.path.join(
            out_dir, 'product.json')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'cc.nii.gz')))
        assert_true(os.stat(os.path.join(out_dir, 'cc.nii.gz')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'mask_noise.nii.gz')))
        assert_true(os.stat(os.path.join(
            out_dir, 'mask_noise.nii.gz')).st_size != 0)

        snr_flow._force_overwrite = True
        snr_flow.run(*args, out_dir=out_dir)
        assert_true(os.path.exists(os.path.join(out_dir, 'product.json')))
        assert_true(os.stat(os.path.join(
            out_dir, 'product.json')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'cc.nii.gz')))
        assert_true(os.stat(os.path.join(out_dir, 'cc.nii.gz')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'mask_noise.nii.gz')))
        assert_true(os.stat(os.path.join(
            out_dir, 'mask_noise.nii.gz')).st_size != 0)

        snr_flow._force_overwrite = True
        snr_flow.run(*args, bbox_threshold=(0.5, 1, 0,
                                            0.15, 0, 0.2), out_dir=out_dir)
        assert_true(os.path.exists(os.path.join(out_dir, 'product.json')))
        assert_true(os.stat(os.path.join(
            out_dir, 'product.json')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'cc.nii.gz')))
        assert_true(os.stat(os.path.join(out_dir, 'cc.nii.gz')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'mask_noise.nii.gz')))
        assert_true(os.stat(os.path.join(
            out_dir, 'mask_noise.nii.gz')).st_size != 0)


@pytest.mark.skipif(not have_pandas or not have_statsmodels or not have_tables,
                    reason='Requires Pandas, StatsModels and PyTables')
def test_bundle_analysis_population_flow():

    with TemporaryDirectory() as dirpath:
        data_path = get_fnames('fornix')
        fornix = load_tractogram(data_path, 'same',
                                 bbox_valid_check=False).streamlines

        f = Streamlines(fornix)

        mb = os.path.join(dirpath, "model_bundles")
        sub = os.path.join(dirpath, "subjects")

        os.mkdir(mb)
        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(sft, os.path.join(mb, "temp.trk"),
                        bbox_valid_check=False)

        os.mkdir(sub)

        os.mkdir(os.path.join(sub, "patient"))

        os.mkdir(os.path.join(sub, "control"))

        p = os.path.join(sub, "patient", "10001")
        os.mkdir(p)

        c = os.path.join(sub, "control", "20002")
        os.mkdir(c)

        for pre in [p, c]:

            os.mkdir(os.path.join(pre, "rec_bundles"))

            sft = StatefulTractogram(f, data_path, Space.RASMM)
            save_tractogram(sft, os.path.join(pre, "rec_bundles", "temp.trk"),
                            bbox_valid_check=False)
            os.mkdir(os.path.join(pre, "org_bundles"))

            sft = StatefulTractogram(f, data_path, Space.RASMM)
            save_tractogram(sft, os.path.join(pre, "org_bundles", "temp.trk"),
                            bbox_valid_check=False)
            os.mkdir(os.path.join(pre, "measures"))

            fa = np.random.rand(255, 255, 255)

            save_nifti(os.path.join(pre, "measures", "fa.nii.gz"),
                       fa, affine=np.eye(4))

        out_dir = os.path.join(dirpath, "output")
        os.mkdir(out_dir)

        ba_flow = BundleAnalysisPopulationFlow()

        ba_flow.run(mb, sub, out_dir=out_dir)

        assert_true(os.path.exists(os.path.join(out_dir, 'fa.h5')))

        dft = pd.read_hdf(os.path.join(out_dir, 'fa.h5'))

        assert_true(dft.bundle.unique() == "temp")

        assert_true(set(dft.subject.unique()) == set(['10001', '20002']))


@pytest.mark.skipif(not have_pandas or not have_statsmodels or not have_tables,
                    reason='Requires Pandas, StatsModels and PyTables')
def test_linear_mixed_models_flow():

    with TemporaryDirectory() as dirpath:

        out_dir = os.path.join(dirpath, "output")
        os.mkdir(out_dir)

        d = {'bundle': ["temp"]*100,
             'disk#': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]*10,
             'fa': [0.21, 0.234, 0.44, 0.44, 0.5, 0.23, 0.55, 0.34, 0.76,
                    0.34]*10,
             'subject': ["10001", "10001", "10001", "10001", "10001",
                         "20002", "20002", "20002", "20002", "20002"]*10,
             'group': ["control", "control", "control", "control", "control",
                       "patient", "patient", "patient", "patient",
                       "patient"]*10}

        df = pd.DataFrame(data=d)
        store = pd.HDFStore(os.path.join(out_dir, 'fa.h5'))
        store.append('fa', df, data_columns=True)
        store.close()

        lmm_flow = LinearMixedModelsFlow()

        out_dir2 = os.path.join(dirpath, "output2")
        os.mkdir(out_dir2)

        input_path = os.path.join(out_dir, "*")

        lmm_flow.run(input_path, no_disks=5, out_dir=out_dir2)

        assert_true(os.path.exists(os.path.join(out_dir2, 'temp_fa.png')))

        # test error
        d2 = {'bundle': ["temp"]*10,
              'disk#': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]*1,
              'fa': [0.21, 0.234, 0.44, 0.44, 0.5, 0.23, 0.55, 0.34, 0.76,
                     0.34]*1,
              'subject': ["10001", "10001", "10001", "10001", "10001",
                          "20002", "20002", "20002", "20002", "20002"]*1,
              'group': ["control", "control", "control", "control", "control",
                        "patient", "patient", "patient", "patient",
                        "patient"]*1}

        df = pd.DataFrame(data=d2)

        out_dir3 = os.path.join(dirpath, "output3")
        os.mkdir(out_dir3)

        store = pd.HDFStore(os.path.join(out_dir3, 'fa.h5'))
        store.append('fa', df, data_columns=True)
        store.close()

        out_dir4 = os.path.join(dirpath, "output4")
        os.mkdir(out_dir4)

        input_path = os.path.join(out_dir3, "*")

        npt.assert_raises(ValueError, lmm_flow.run, input_path, no_disks=5,
                          out_dir=out_dir4)


if __name__ == '__main__':
    npt.run_module_suite()
