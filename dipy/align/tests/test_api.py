import os.path as op

import numpy as np
import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.data as dpd
import dipy.core.gradients as dpg

from dipy.align import (syn_registration, register_series, register_dwi_series,
                        c_of_mass, translation, rigid, affine,
                        streamline_registration, write_mapping,
                        read_mapping, dwi_to_template)

from dipy.align.imwarp import DiffeomorphicMap

from dipy.tracking.utils import transform_tracking_output
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import StatefulTractogram, Space

MNI_T2 = dpd.read_mni_template()
hardi_img, gtab = dpd.read_stanford_hardi()
MNI_T2_data = MNI_T2.get_fdata()
MNI_T2_affine = MNI_T2.affine
hardi_data = hardi_img.get_fdata()
hardi_affine = hardi_img.affine
b0 = hardi_data[..., gtab.b0s_mask]
mean_b0 = np.mean(b0, -1)

# We select some arbitrary chunk of data so this goes quicker:
subset_b0 = mean_b0[40:50, 40:50, 40:50]
subset_dwi_data = nib.Nifti1Image(hardi_data[40:50, 40:50, 40:50],
                                  hardi_affine)
subset_t2 = MNI_T2_data[40:60, 40:60, 40:60]
subset_b0_img = nib.Nifti1Image(subset_b0, hardi_affine)
subset_t2_img = nib.Nifti1Image(subset_t2, MNI_T2_affine)


def test_syn_registration():
    with nbtmp.InTemporaryDirectory() as tmpdir:
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
            assert (np.all(mapping.__getattribute__(k) ==
                           file_mapping.__getattribute__(k)))


def test_dwi_to_template():
    warped_b0, mapping = dwi_to_template(subset_dwi_data, gtab,
                                          template=subset_t2_img,
                                          radius=1)
    npt.assert_equal(isinstance(mapping, DiffeomorphicMap), True)
    npt.assert_equal(warped_b0.shape, subset_t2_img.shape)


def test_register_series():
    fdata, fbval, fbvec = dpd.get_fnames('small_64D')
    img = nib.load(fdata)
    gtab = dpg.gradient_table(fbval, fbvec)
    ref_idx = np.where(gtab.b0s_mask)[0][0]
    xformed, affines = register_series(img, ref_idx)
    assert np.all(affines[..., ref_idx] == np.eye(4))
    assert np.all(xformed[..., ref_idx] == img.get_fdata()[..., ref_idx])


def test_register_dwi_series():
    fdata, fbval, fbvec = dpd.get_fnames('small_64D')
    with nbtmp.InTemporaryDirectory() as tmpdir:
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
        npt.assert_(isinstance(reg_img, nib.Nifti1Image))


def test_streamline_registration():
    sl1 = [np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]]),
           np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 1, 1]])]
    affine = np.eye(4)
    affine[:3, 3] = np.random.randn(3)
    sl2 = list(transform_tracking_output(sl1, affine))
    aligned, matrix = streamline_registration(sl2, sl1)
    npt.assert_almost_equal(matrix, np.linalg.inv(affine))
    npt.assert_almost_equal(aligned[0], sl1[0])
    npt.assert_almost_equal(aligned[1], sl1[1])

    # We assume the two tracks come from the same space, but it might have
    # some affine associated with it:
    base_aff = np.eye(4) * np.random.rand()
    base_aff[:3, 3] = np.array([1, 2, 3])
    base_aff[3, 3] = 1

    with nbtmp.InTemporaryDirectory() as tmpdir:
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