import numpy.testing as npt

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

from dipy.data import get_data
from dipy.workflows.align import ResliceFlow

import os.path
from os.path import join as pjoin
import shutil

from dipy.align.tests.test_parzenhist import setup_random_transform
from dipy.align.transforms import (Transform,
                                   regtransforms)
from dipy.io.image import save_nifti
from glob import glob

from dipy.workflows.align import ImageRegistrationFlow


def test_reslice():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_data('small_25')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()

        reslice_flow = ResliceFlow()
        reslice_flow.run(data_path, [1.5, 1.5, 1.5], out_dir=out_dir)

        out_path = reslice_flow.last_generated_outputs['out_resliced']
        out_img = nib.load(out_path)
        resliced = out_img.get_data()

        npt.assert_equal(resliced.shape[0] > volume.shape[0], True)
        npt.assert_equal(resliced.shape[1] > volume.shape[1], True)
        npt.assert_equal(resliced.shape[2] > volume.shape[2], True)
        npt.assert_equal(resliced.shape[-1], volume.shape[-1])


def test_image_registration():

    with TemporaryDirectory() as temp_out_dir:
        static, moving, static_g2w, moving_g2w, smask, mmask, M = setup_random_transform(
            transform=regtransforms[('AFFINE', 3)], rfactor=0.1)

        save_nifti(pjoin(temp_out_dir, 'b0.nii.gz'), data=static,
                   affine=static_g2w)
        save_nifti(pjoin(temp_out_dir, 't1.nii.gz'), data=moving,
                   affine=moving_g2w)

        static_image_file = pjoin(temp_out_dir, 'b0.nii.gz')
        moving_image_file = pjoin(temp_out_dir, 't1.nii.gz')

        out_moved = pjoin(temp_out_dir, "com_moved.nii.gz")
        out_affine = pjoin(temp_out_dir, "com_affine.txt")

        image_registeration_flow = ImageRegistrationFlow()

        image_registeration_flow.run(static_image_file,
                                     moving_image_file,
                                     transform='com',
                                     out_dir=temp_out_dir
                                     , out_moved=out_moved,
                                     out_affine=out_affine,
                                     level_iters="100 10 1")

        npt.assert_equal(os.path.exists(out_moved), True)
        npt.assert_equal(os.path.exists(out_affine), True)

        out_moved = pjoin(temp_out_dir, "trans_moved.nii.gz")
        out_affine = pjoin(temp_out_dir, "trans_affine.txt")

        image_registeration_flow.run(static_image_file,
                                     moving_image_file,
                                     transform='trans',
                                     out_dir=temp_out_dir
                                     , out_moved=out_moved,
                                     out_affine=out_affine,
                                     level_iters="100 10 1")

        npt.assert_equal(os.path.exists(out_moved), True)
        npt.assert_equal(os.path.exists(out_affine), True)

        out_moved = pjoin(temp_out_dir, "rigid_moved.nii.gz")
        out_affine = pjoin(temp_out_dir, "rigid_affine.txt")

        image_registeration_flow.run(static_image_file,
                                     moving_image_file,
                                     transform='rigid',
                                     out_dir=temp_out_dir
                                     , out_moved=out_moved,
                                     out_affine=out_affine,
                                     level_iters="100 10 1")

        npt.assert_equal(os.path.exists(out_moved), True)
        npt.assert_equal(os.path.exists(out_affine), True)

        out_moved = pjoin(temp_out_dir, "affine_moved.nii.gz")
        out_affine = pjoin(temp_out_dir, "affine_affine.txt")

        image_registeration_flow.run(static_image_file,
                                     moving_image_file,
                                     transform='affine',
                                     out_dir=temp_out_dir
                                     , out_moved=out_moved,
                                     out_affine=out_affine,
                                     level_iters="100 10 1")

        npt.assert_equal(os.path.exists(out_moved), True)
        npt.assert_equal(os.path.exists(out_affine), True)

        # Creating the erroneous behavior
        npt.assert_raises(
            ValueError, image_registeration_flow.run,
            static_image_file,
            moving_image_file,
            transform='notransform')

        npt.assert_raises(
            ValueError, image_registeration_flow.run,
            static_image_file,
            moving_image_file,
            metric='wrong_metric')


# Uncomment for manual debugging
            #copy_output(temp_out_dir)


# def copy_output(temp_directory_path):
# set the folder_path to the directory where the registered images will be copied.
#     folder_path = ''
#     out_files = list(glob(pjoin(temp_directory_path, '*.nii.gz')) + glob(pjoin(temp_directory_path, '*.txt')))
#
#     for out_file in out_files:
#         shutil.copy(out_file, folder_path)


if __name__ == '__main__':
    test_reslice()
    test_image_registration()