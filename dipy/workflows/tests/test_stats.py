#!/usr/bin/env python3

import os
from os.path import join
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.io.image import save_nifti, load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.testing import assert_true
from dipy.testing.decorators import set_random_number_generator
from dipy.tracking.streamline import Streamlines
from dipy.utils.optpkg import optional_package
from dipy.workflows.stats import SNRinCCFlow
from dipy.workflows.stats import BundleAnalysisTractometryFlow
from dipy.workflows.stats import LinearMixedModelsFlow
from dipy.workflows.stats import BundleShapeAnalysis
from dipy.workflows.stats import buan_bundle_profiles

pd, have_pandas, _ = optional_package("pandas")
_, have_statsmodels, _ = optional_package("statsmodels", min_version="0.14.0")
_, have_tables, _ = optional_package("tables")
_, have_matplotlib, _ = optional_package("matplotlib")


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


@pytest.mark.skipif(not have_pandas or not have_statsmodels or not have_tables
                    or not have_matplotlib,
                    reason='Requires Pandas, StatsModels and PyTables')
@set_random_number_generator()
def test_buan_bundle_profiles(rng):

    with TemporaryDirectory() as dirpath:
        data_path = get_fnames('fornix')
        fornix = load_tractogram(data_path, 'same',
                                 bbox_valid_check=False).streamlines

        f = Streamlines(fornix)

        mb = os.path.join(dirpath, "model_bundles")

        os.mkdir(mb)

        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(sft, os.path.join(mb, "temp.trk"),
                        bbox_valid_check=False)

        rb = os.path.join(dirpath, "rec_bundles")
        os.mkdir(rb)

        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(sft, os.path.join(rb, "temp.trk"),
                        bbox_valid_check=False)

        ob = os.path.join(dirpath, "org_bundles")
        os.mkdir(ob)

        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(sft, os.path.join(ob, "temp.trk"),
                        bbox_valid_check=False)

        dt = os.path.join(dirpath, "anatomical_measures")
        os.mkdir(dt)

        fa = rng.random((255, 255, 255))

        save_nifti(os.path.join(dt, "fa.nii.gz"),
                   fa, affine=np.eye(4))

        out_dir = os.path.join(dirpath, "output")
        os.mkdir(out_dir)

        buan_bundle_profiles(mb, rb, ob, dt, group_id=1, subject="10001",
                             no_disks=100, out_dir=out_dir)

        assert_true(os.path.exists(os.path.join(out_dir, 'temp_fa.h5')))


@pytest.mark.skipif(not have_pandas or not have_statsmodels or not have_tables
                    or not have_matplotlib,
                    reason='Requires Pandas, StatsModels, PyTables, and '
                           'matplotlib')
@set_random_number_generator()
def test_bundle_analysis_tractometry_flow(rng):

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
            os.mkdir(os.path.join(pre, "anatomical_measures"))

            fa = rng.random((255, 255, 255))

            save_nifti(os.path.join(pre, "anatomical_measures", "fa.nii.gz"),
                       fa, affine=np.eye(4))

        out_dir = os.path.join(dirpath, "output")
        os.mkdir(out_dir)

        ba_flow = BundleAnalysisTractometryFlow()

        ba_flow.run(mb, sub, out_dir=out_dir)

        assert_true(os.path.exists(os.path.join(out_dir, 'temp_fa.h5')))

        dft = pd.read_hdf(os.path.join(out_dir, 'temp_fa.h5'))

        # assert_true(dft.bundle.unique() == "temp")

        assert_true(set(dft.subject.unique()) == {'10001', '20002'})


@pytest.mark.skipif(not have_pandas or not have_statsmodels or not have_tables
                    or not have_matplotlib,
                    reason='Requires Pandas, StatsModels, PyTables, and '
                           'matplotlib')
def test_linear_mixed_models_flow():

    with TemporaryDirectory() as dirpath:

        out_dir = os.path.join(dirpath, "output")
        os.mkdir(out_dir)

        d = {'disk': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]*10,
             'fa': [0.21, 0.234, 0.44, 0.44, 0.5, 0.23, 0.55, 0.34, 0.76,
                    0.34]*10,
             'subject': ["10001", "10001", "10001", "10001", "10001",
                         "20002", "20002", "20002", "20002", "20002"]*10,
             'group': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]*10}

        df = pd.DataFrame(data=d)
        store = pd.HDFStore(os.path.join(out_dir, 'temp_fa.h5'))
        store.append('fa', df, data_columns=True)
        store.close()

        lmm_flow = LinearMixedModelsFlow()

        out_dir2 = os.path.join(dirpath, "output2")
        os.mkdir(out_dir2)

        input_path = os.path.join(out_dir, "*")

        lmm_flow.run(input_path, no_disks=5, out_dir=out_dir2)

        assert_true(os.path.exists(os.path.join(out_dir2,
                                                'temp_fa_pvalues.npy')))
        assert_true(os.path.exists(os.path.join(out_dir2,
                                                'temp_fa.png')))

        # test error
        d2 = {'disk': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]*1,
              'fa': [0.21, 0.234, 0.44, 0.44, 0.5, 0.23, 0.55, 0.34, 0.76,
                     0.34]*1,
              'subject': ["10001", "10001", "10001", "10001", "10001",
                          "20002", "20002", "20002", "20002", "20002"]*1,
              'group': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]*1}

        df = pd.DataFrame(data=d2)

        out_dir3 = os.path.join(dirpath, "output3")
        os.mkdir(out_dir3)

        store = pd.HDFStore(os.path.join(out_dir3, 'temp_fa.h5'))
        store.append('fa', df, data_columns=True)
        store.close()

        out_dir4 = os.path.join(dirpath, "output4")
        os.mkdir(out_dir4)

        input_path = os.path.join(out_dir3, "f*")
        # OS error raised if path is wrong or file does not exist
        npt.assert_raises(OSError, lmm_flow.run, input_path, no_disks=5,
                          out_dir=out_dir4)

        input_path = os.path.join(out_dir3, "*")
        # value error raised if length of data frame is less than 100
        npt.assert_raises(ValueError, lmm_flow.run, input_path, no_disks=5,
                          out_dir=out_dir4)


@pytest.mark.skipif(not have_pandas or not have_statsmodels or not have_tables
                    or not have_matplotlib,
                    reason='Requires Pandas, StatsModels, PyTables, and '
                           'matplotlib')
@set_random_number_generator()
def test_bundle_shape_analysis_flow(rng):

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
            os.mkdir(os.path.join(pre, "anatomical_measures"))

            fa = rng.random((255, 255, 255))

            save_nifti(os.path.join(pre, "anatomical_measures", "fa.nii.gz"),
                       fa, affine=np.eye(4))

        out_dir = os.path.join(dirpath, "output")
        os.mkdir(out_dir)

        sm_flow = BundleShapeAnalysis()

        sm_flow.run(sub, out_dir=out_dir)

        assert_true(os.path.exists(os.path.join(out_dir, "temp.npy")))
