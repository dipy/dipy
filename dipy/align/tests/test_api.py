import os.path as op
import pytest
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as npt

import nibabel as nib

import dipy.data as dpd
import dipy.core.gradients as dpg

from dipy.align import (syn_registration, register_series, register_dwi_series,
                        center_of_mass, translation, rigid_isoscaling,
                        rigid_scaling, rigid, affine, motion_correction,
                        affine_registration, streamline_registration,
                        write_mapping, read_mapping, register_dwi_to_template)

from dipy.align.imwarp import DiffeomorphicMap

from dipy.tracking.utils import transform_tracking_output
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.testing.decorators import set_random_number_generator


def setup_module():
    global subset_b0, subset_dwi_data, subset_t2, subset_b0_img, \
           subset_t2_img, gtab, hardi_affine, MNI_T2_affine
    MNI_T2 = dpd.read_mni_template()
    hardi_img, gtab = dpd.read_stanford_hardi()
    MNI_T2_data = MNI_T2.get_fdata()
    MNI_T2_affine = MNI_T2.affine
    hardi_data = hardi_img.get_fdata()
    hardi_affine = hardi_img.affine
    b0 = hardi_data[..., gtab.b0s_mask]
    mean_b0 = np.mean(b0, -1)

    # We select some arbitrary chunk of data so this goes quicker:
    subset_b0 = mean_b0[40:45, 40:45, 40:45]
    subset_dwi_data = nib.Nifti1Image(hardi_data[40:45, 40:45, 40:45],
                                      hardi_affine)
    subset_t2 = MNI_T2_data[40:50, 40:50, 40:50]
    subset_b0_img = nib.Nifti1Image(subset_b0, hardi_affine)
    subset_t2_img = nib.Nifti1Image(subset_t2, MNI_T2_affine)


def test_syn_registration():
    with TemporaryDirectory() as tmpdir:
        warped_moving, mapping = syn_registration(subset_b0,
                                                  subset_t2,
                                                  moving_affine=hardi_affine,
                                                  static_affine=MNI_T2_affine,
                                                  step_length=0.1,
                                                  metric='CC',
                                                  dim=3,
                                                  level_iters=[5, 5, 5],
                                                  sigma_diff=2.0,
                                                  radius=1,
                                                  prealign=None)

        npt.assert_equal(warped_moving.shape, subset_t2.shape)
        mapping_fname = op.join(tmpdir, 'mapping.nii.gz')
        write_mapping(mapping, mapping_fname)
        file_mapping = read_mapping(mapping_fname,
                                    subset_b0_img,
                                    subset_t2_img)

        # Test that it has the same effect on the data:
        warped_from_file = file_mapping.transform(subset_b0)
        npt.assert_equal(warped_from_file, warped_moving)

        # Test that it is, attribute by attribute, identical:
        for k in mapping.__dict__:
            npt.assert_((np.all(mapping.__getattribute__(k) ==
                                file_mapping.__getattribute__(k))))


def test_register_dwi_to_template():
    # Default is syn registration:
    warped_b0, mapping = register_dwi_to_template(subset_dwi_data, gtab,
                                                  template=subset_t2_img,
                                                  level_iters=[5, 5, 5],
                                                  sigma_diff=2.0,
                                                  radius=1)
    npt.assert_(isinstance(mapping, DiffeomorphicMap))
    npt.assert_equal(warped_b0.shape, subset_t2_img.shape)

    # Use affine registration (+ don't provide a template and inputs as
    # strings):
    fdata, fbval, fbvec = dpd.get_fnames('small_64D')
    warped_data, affine_mat = register_dwi_to_template(fdata, (fbval, fbvec),
                                                       reg_method="aff",
                                                       level_iters=[5, 5, 5],
                                                       sigmas=[3, 1, 0],
                                                       factors=[4, 2, 1])
    npt.assert_(isinstance(affine_mat, np.ndarray))
    npt.assert_(affine_mat.shape == (4, 4))


def test_affine_registration():
    moving = subset_b0
    static = subset_b0
    moving_affine = static_affine = np.eye(4)
    xformed, affine_mat = affine_registration(moving, static,
                                              moving_affine=moving_affine,
                                              static_affine=static_affine,
                                              level_iters=[5, 5],
                                              sigmas=[3, 1],
                                              factors=[2, 1])

    # We don't ask for much:
    npt.assert_almost_equal(affine_mat[:3, :3], np.eye(3), decimal=1)

    # [center_of_mass] + ret_metric=True should raise an error
    with pytest.raises(ValueError):
        # For array input, must provide affines:
        xformed, affine_mat = affine_registration(moving, static,
                                                  moving_affine=moving_affine,
                                                  static_affine=static_affine,
                                                  pipeline=["center_of_mass"],
                                                  ret_metric=True)

    # Define list of methods
    reg_methods = ["center_of_mass", "translation", "rigid",
                   "rigid_isoscaling", "rigid_scaling", "affine",
                   center_of_mass, translation, rigid,
                   rigid_isoscaling, rigid_scaling, affine]

    # Test methods individually (without returning any metric)
    for func in reg_methods:
        xformed, affine_mat = affine_registration(moving, static,
                                                  moving_affine=moving_affine,
                                                  static_affine=static_affine,
                                                  level_iters=[5, 5],
                                                  sigmas=[3, 1],
                                                  factors=[2, 1],
                                                  pipeline=[func])
        # We don't ask for much:
        npt.assert_almost_equal(affine_mat[:3, :3], np.eye(3), decimal=1)

    # Bad method
    with pytest.raises(ValueError, match=r'^pipeline\[0\] must be one.*foo.*'):
        affine_registration(
            moving, static, moving_affine, static_affine, pipeline=['foo'])

    # Test methods individually (returning quality metric)
    expected_nparams = [0, 3, 6, 7, 9, 12] * 2
    assert len(expected_nparams) == len(reg_methods)
    for i, func in enumerate(reg_methods):
        if func in ('center_of_mass', center_of_mass):
            # can't return metric
            with pytest.raises(ValueError, match='cannot return any quality'):
                affine_registration(
                    moving, static, moving_affine, static_affine,
                    pipeline=[func], ret_metric=True)
            continue

        xformed, affine_mat, \
            xopt, fopt = affine_registration(moving, static,
                                             moving_affine=moving_affine,
                                             static_affine=static_affine,
                                             level_iters=[5, 5],
                                             sigmas=[3, 1],
                                             factors=[2, 1],
                                             pipeline=[func],
                                             ret_metric=True)
        # Expected number of optimization parameters
        npt.assert_equal(len(xopt), expected_nparams[i])
        # Optimization metric must be a single numeric value
        npt.assert_equal(isinstance(fopt, (int, float)), True)

    with pytest.raises(ValueError):
        # For array input, must provide affines:
        xformed, affine_mat = affine_registration(moving, static)

    # Not supported transform names should raise an error
    npt.assert_raises(ValueError, affine_registration, moving, static,
                      moving_affine, static_affine,
                      pipeline=["wrong_transform"])

    # If providing nifti image objects, don't need to provide affines:
    moving_img = nib.Nifti1Image(moving, moving_affine)
    static_img = nib.Nifti1Image(static, static_affine)
    xformed, affine_mat = affine_registration(moving_img, static_img)
    npt.assert_almost_equal(affine_mat[:3, :3], np.eye(3), decimal=1)

    # Using strings with full paths as inputs also works:
    t1_name, b0_name = dpd.get_fnames('syn_data')
    moving = b0_name
    static = t1_name
    xformed, affine_mat = affine_registration(moving, static,
                                              level_iters=[5, 5],
                                              sigmas=[3, 1],
                                              factors=[4, 2])
    npt.assert_almost_equal(affine_mat[:3, :3], np.eye(3), decimal=1)


def test_single_transforms():
    moving = subset_b0
    static = subset_b0
    moving_affine = static_affine = np.eye(4)

    reg_methods = [center_of_mass, translation, rigid_isoscaling,
                   rigid_scaling, rigid, affine]

    for func in reg_methods:
        xformed, affine_mat = func(moving, static, moving_affine,
                                   static_affine, level_iters=[5, 5],
                                   sigmas=[3, 1], factors=[2, 1])
        # We don't ask for much:
        npt.assert_almost_equal(affine_mat[:3, :3], np.eye(3), decimal=1)


def test_register_series():
    fdata, fbval, fbvec = dpd.get_fnames('small_64D')
    img = nib.load(fdata)
    gtab = dpg.gradient_table(fbval, fbvec)
    ref_idx = np.where(gtab.b0s_mask)[0][0]
    xformed, affines = register_series(img, ref_idx)
    npt.assert_(np.all(affines[..., ref_idx] == np.eye(4)))
    npt.assert_(np.all(xformed[..., ref_idx] == img.get_fdata()[..., ref_idx]))


def test_register_dwi_series_and_motion_correction():
    fdata, fbval, fbvec = dpd.get_fnames('small_64D')
    with TemporaryDirectory() as tmpdir:
        # Use an abbreviated data-set:
        img = nib.load(fdata)
        data = img.get_fdata()[..., :10]
        nib.save(nib.Nifti1Image(data, img.affine),
                 op.join(tmpdir, 'data.nii.gz'))
        # Save a subset:
        bvals = np.loadtxt(fbval)
        bvecs = np.loadtxt(fbvec)
        np.savetxt(op.join(tmpdir, 'bvals.txt'), bvals[:10])
        np.savetxt(op.join(tmpdir, 'bvecs.txt'), bvecs[:10])
        gtab = dpg.gradient_table(op.join(tmpdir, 'bvals.txt'),
                                  op.join(tmpdir, 'bvecs.txt'))
        reg_img, reg_affines = register_dwi_series(data, gtab, img.affine)
        reg_img_2, reg_affines_2 = motion_correction(data, gtab, img.affine)
        npt.assert_(isinstance(reg_img, nib.Nifti1Image))

        npt.assert_array_equal(reg_img.get_fdata(), reg_img_2.get_fdata())
        npt.assert_array_equal(reg_affines, reg_affines_2)


@set_random_number_generator()
def test_streamline_registration(rng):
    sl1 = [np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]]),
           np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 1, 1]])]
    affine_mat = np.eye(4)
    affine_mat[:3, 3] = rng.standard_normal(3)
    sl2 = list(transform_tracking_output(sl1, affine_mat))
    aligned, matrix = streamline_registration(sl2, sl1)
    npt.assert_almost_equal(matrix, np.linalg.inv(affine_mat))
    npt.assert_almost_equal(aligned[0], sl1[0])
    npt.assert_almost_equal(aligned[1], sl1[1])

    # We assume the two tracks come from the same space, but it might have
    # some affine associated with it:
    base_aff = np.eye(4) * rng.random()
    base_aff[:3, 3] = np.array([1, 2, 3])
    base_aff[3, 3] = 1

    with TemporaryDirectory() as tmpdir:
        for use_aff in [None, base_aff]:
            fname1 = op.join(tmpdir, 'sl1.trk')
            fname2 = op.join(tmpdir, 'sl2.trk')
            if use_aff is not None:
                img = nib.Nifti1Image(np.zeros((2, 2, 2)), use_aff)
                # Move the streamlines to this other space, and report it:
                tgm1 = StatefulTractogram(
                    transform_tracking_output(sl1, np.linalg.inv(use_aff)),
                    img,
                    Space.VOX)

                save_trk(tgm1, fname1, bbox_valid_check=False)

                tgm2 = StatefulTractogram(
                    transform_tracking_output(sl2, np.linalg.inv(use_aff)),
                    img,
                    Space.VOX)

                save_trk(tgm2, fname2, bbox_valid_check=False)

            else:
                img = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
                tgm1 = StatefulTractogram(sl1, img, Space.RASMM)
                tgm2 = StatefulTractogram(sl2, img, Space.RASMM)
                save_trk(tgm1, fname1, bbox_valid_check=False)
                save_trk(tgm2, fname2, bbox_valid_check=False)

            aligned, matrix = streamline_registration(fname2, fname1)
            npt.assert_almost_equal(aligned[0], sl1[0], decimal=5)
            npt.assert_almost_equal(aligned[1], sl1[1], decimal=5)


def test_register_dwi_series_multi_b0():
    # Test if register_dwi_series works with multiple b0 images
    dwi_fname, dwi_bval_fname, \
        dwi_bvec_fname = dpd.get_fnames('sherbrooke_3shell')
    data, affine = load_nifti(dwi_fname)
    bvals, bvecs = read_bvals_bvecs(dwi_bval_fname, dwi_bvec_fname)

    data_small = data[..., :3]
    data_small = np.concatenate([data[..., :1], data_small], axis=-1)
    bvals_small = np.concatenate([bvals[:1], bvals[:3]], axis=0)
    bvecs_small = np.concatenate([bvecs[:1], bvecs[:3]], axis=0)
    gtab = dpg.gradient_table(bvals_small, bvecs_small)
    _ = motion_correction(data_small, gtab, affine)
