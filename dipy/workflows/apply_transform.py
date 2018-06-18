from dipy.workflows.workflow import Workflow

from dipy.align.imaffine import AffineMap
import nibabel as nib
import numpy as np
from os.path import join as pjoin
from os import path
from dipy.io.image import save_nifti, load_affine_matrix


class ApplyTransformFlow(Workflow):

    def run(self, reference_image_file, moving_image_files, affine_matrix_file,
            out_file='transformed.nii.gz'):

        """
        Parameters
        ----------
        reference_image_file : string
            Path to the static image file.

        moving_image_files : string
            Location of the file or folder containing the images to be transformed.

        affine_matrix_file : string
            The text file containing the affine matrix for transformation.

        out_file : string, optional
            Name of the transformed file. (default 'transformed.nii.gz')
        """

        io = self.get_io_iterator()

        for ref_image_file, mov_images, affine_matrix_file, out_file in io:

            ref_image = nib.load(ref_image_file)
            static_grid2world = ref_image.affine

            image = nib.load(mov_images)
            image_data = np.array(image.get_data())

            affine_matrix = load_affine_matrix(affine_matrix_file)

            img_transformation = AffineMap(affine=affine_matrix, domain_grid_shape=image_data.shape)
            transformed = img_transformation.transform(image_data)

            save_nifti(pjoin(path.split(mov_images)[1]+out_file), transformed, affine=static_grid2world)