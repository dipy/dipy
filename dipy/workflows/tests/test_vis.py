#!python

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory
from numpy.testing import run_module_suite
import os.path
from os.path import join as pjoin

from dipy.align.tests.test_parzenhist import setup_random_transform
from dipy.align.transforms import Transform, regtransforms
from dipy.io.image import save_nifti
from dipy.workflows.align import ImageRegistrationFlow
from dipy.workflows.vis_registration import VisualizeRegisteredImage


def test_mosaic():

    with TemporaryDirectory() as temp_out_dir:

        static, moving, static_g2w, moving_g2w, smask, mmask, M \
            = setup_random_transform(transform=regtransforms[('AFFINE', 3)],
                                     rfactor=0.1)

        save_nifti(pjoin(temp_out_dir, 'b0.nii.gz'), data=static,
                   affine=static_g2w)
        save_nifti(pjoin(temp_out_dir, 't1.nii.gz'), data=moving,
                   affine=moving_g2w)

        static_image_file = pjoin(temp_out_dir, 'b0.nii.gz')
        moving_image_file = pjoin(temp_out_dir, 't1.nii.gz')

        image_registeration_flow = ImageRegistrationFlow()
        vis_registered = VisualizeRegisteredImage()

        out_moved = pjoin(temp_out_dir, "affine_moved.nii.gz")
        out_affine = pjoin(temp_out_dir, "affine_affine.txt")

        image_registeration_flow.run(static_image_file,
                                     moving_image_file,
                                     transform='affine',
                                     out_dir=temp_out_dir,
                                     out_moved=out_moved,
                                     out_affine=out_affine,
                                     save_metric=False,
                                     level_iters=[100, 10, 1],
                                     out_quality='affine_q.txt')

        moved_data = nib.load(out_moved)
        moved_img = moved_data.get_data()
        moving_grid2wordld = moved_data.affine

        overlapped = vis_registered.create_mosaic(static,
                                                  moved_img, moving_grid2wordld, fname=None)

        x, y, z = overlapped.shape[:3]

        assert overlapped[x//2, :, :, 2] == 0
        assert overlapped[:, y//2, :, 2] == 0
        assert overlapped[:, :, z//2, 2] == 0

if __name__ == "__main__":
    run_module_suite()





