import numpy.testing as npt
import numpy as np


from time import sleep
import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

from dipy.data import get_data
from dipy.workflows.align import ResliceFlow

import os.path
from os.path import join as pjoin

from dipy.align.tests.test_parzenhist import setup_random_transform
from dipy.align.transforms import (Transform,
                                   regtransforms)
from dipy.io.image import save_nifti

from dipy.workflows.align import ImageRegistrationFlow
from dipy.workflows.align import ApplyTransformFlow


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
        static, moving, static_g2w, moving_g2w, sm, mm, m = \
            setup_random_transform(transform=regtransforms[('AFFINE', 3)],
                                   rfactor=0.1)

        save_nifti(pjoin(temp_out_dir, 'b0.nii.gz'), data=static,
                   affine=static_g2w)
        save_nifti(pjoin(temp_out_dir, 't1.nii.gz'), data=moving,
                   affine=moving_g2w)

        static_image_file = pjoin(temp_out_dir, 'b0.nii.gz')
        moving_image_file = pjoin(temp_out_dir, 't1.nii.gz')

        image_registeration_flow = ImageRegistrationFlow()

        def read_distance(qual_fname):
            temp_val = 0
            with open(pjoin(temp_out_dir, qual_fname), 'r') as f:
                for line in f:
                    pass
                temp_val = line
            return float(temp_val)

        def test_com():

            out_moved = pjoin(temp_out_dir, "com_moved.nii.gz")
            out_affine = pjoin(temp_out_dir, "com_affine.txt")

            image_registeration_flow.run(static_image_file,
                                         moving_image_file,
                                         transform='com',
                                         out_dir=temp_out_dir,
                                         out_moved=out_moved,
                                         out_affine=out_affine)

            npt.assert_equal(os.path.exists(out_moved), True)
            npt.assert_equal(os.path.exists(out_affine), True)

        def test_translation():

            out_moved = pjoin(temp_out_dir, "trans_moved.nii.gz")
            out_affine = pjoin(temp_out_dir, "trans_affine.txt")

            image_registeration_flow.run(static_image_file,
                                         moving_image_file,
                                         transform='trans',
                                         out_dir=temp_out_dir,
                                         out_moved=out_moved,
                                         out_affine=out_affine,
                                         save_metric=True,
                                         level_iters=[100, 10, 1],
                                         out_quality='trans_q.txt')

            dist = read_distance('trans_q.txt')
            npt.assert_equal('%.2f' % dist, '%.2f' % -0.3953547764454917)
            npt.assert_equal(os.path.exists(out_moved), True)
            npt.assert_equal(os.path.exists(out_affine), True)

        def test_rigid():

            out_moved = pjoin(temp_out_dir, "rigid_moved.nii.gz")
            out_affine = pjoin(temp_out_dir, "rigid_affine.txt")

            image_registeration_flow.run(static_image_file,
                                         moving_image_file,
                                         transform='rigid',
                                         out_dir=temp_out_dir,
                                         out_moved=out_moved,
                                         out_affine=out_affine,
                                         save_metric=True,
                                         level_iters=[100, 10, 1],
                                         out_quality='rigid_q.txt')

            dist = read_distance('rigid_q.txt')
            npt.assert_equal('%.2f' % dist, '%.2f' % -0.6900534794005155)
            npt.assert_equal(os.path.exists(out_moved), True)
            npt.assert_equal(os.path.exists(out_affine), True)

        def test_affine():

            out_moved = pjoin(temp_out_dir, "affine_moved.nii.gz")
            out_affine = pjoin(temp_out_dir, "affine_affine.txt")

            image_registeration_flow.run(static_image_file,
                                         moving_image_file,
                                         transform='affine',
                                         out_dir=temp_out_dir,
                                         out_moved=out_moved,
                                         out_affine=out_affine,
                                         save_metric=True,
                                         level_iters=[100, 10, 1],
                                         out_quality='affine_q.txt')

            dist = read_distance('affine_q.txt')
            npt.assert_equal('%.2f' % dist, '%.2f' % -0.7670650775914811)
            npt.assert_equal(os.path.exists(out_moved), True)
            npt.assert_equal(os.path.exists(out_affine), True)

        # Creating the erroneous behavior
        def test_err():

            npt.assert_raises(ValueError, image_registeration_flow.run,
                              static_image_file,
                              moving_image_file,
                              transform='notransform')

            npt.assert_raises(ValueError, image_registeration_flow.run,
                              static_image_file,
                              moving_image_file,
                              metric='wrong_metric')

        test_com()
        test_translation()
        test_rigid()
        test_affine()
        test_err()


def test_apply_transform():
    with TemporaryDirectory() as temp_out_dir:

        factors = {
            ('TRANSLATION', 3): (2.0, None, np.array([2.3, 4.5, 1.7])),
            ('RIGID', 3): (0.1, None, np.array([0.1, 0.15, -0.11, 2.3, 4.5,
                                                1.7])),
            ('AFFINE', 3): (0.1, None, np.array([0.99, -0.05, 0.03, 1.3,
                                                 0.05, 0.99, -0.10, 2.5,
                                                 -0.07, 0.10, 0.99, -1.4]))}

        for i in factors.keys():
            static, moving, static_g2w, moving_g2w, smask, mmask, M = \
                setup_random_transform(transform=regtransforms[i],
                                       rfactor=factors[i][0])

            some = str(i[0]) + '_static.nii.gz'
            one = str(i[0]) + '_moving.nii.gz'

            save_nifti(pjoin(temp_out_dir, some), data=static,
                       affine=static_g2w)

            save_nifti(pjoin(temp_out_dir, one), data=moving,
                       affine=moving_g2w)

        static_image_file = pjoin(temp_out_dir, 'TRANSLATION_static.nii.gz')
        moving_image_file = pjoin(temp_out_dir, 'TRANSLATION_moving.nii.gz')

        out_moved = pjoin(temp_out_dir, "trans_moved.nii.gz")
        out_affine = pjoin(temp_out_dir, "trans_affine.txt")

        image_registeration_flow = ImageRegistrationFlow()

        image_registeration_flow.run(static_image_file,
                                     moving_image_file,
                                     transform='trans',
                                     out_dir=temp_out_dir,
                                     out_moved=out_moved,
                                     out_affine=out_affine,
                                     level_iters=[100, 10, 1])

        npt.assert_equal(os.path.exists(out_moved), True)
        npt.assert_equal(os.path.exists(out_affine), True)

        apply_trans = ApplyTransformFlow()
        two = pjoin(temp_out_dir, '*moving*')
        apply_trans.run(static_image_file, two, out_affine)

        npt.assert_equal(os.path.exists(pjoin(temp_out_dir,
                                              'transformed.nii.gz')), True)

if __name__ == "__main__":
        test_reslice()
        test_image_registration()
        test_apply_transform()
