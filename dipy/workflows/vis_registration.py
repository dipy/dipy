from __future__ import division, print_function, absolute_import
from dipy.workflows.workflow import Workflow
import logging
import numpy as np
import nibabel as nib
from dipy.workflows.core import write_gif
from dipy.io.image import load_affine_matrix
from dipy.viz.regtools import overlay_slices
from dipy.viz import window, actor

from dipy.viz.window import snapshot


class VisualizeRegisteredImage(Workflow):

    def process_image_data(self, static_img, moved_img):

        """
        Function for pre-processing the image data. It involves
        normalizing and copying static, moving images in the
        red and green channel, respectively.

        Parameters
        ----------
        static_img : array, shape (S, R, C) or (R, C)
            The image to be used as reference during optimization.

        moved_img : array, shape (S', R', C') or (R', C')
            The image to be used as "moving" during optimization.

        Returns
        ------
         overlay : ndarray
            Containing the normalized image data

         value_range : tuple
            A tuple containing the minimum and maximum range of values to
            select the pixels.
        """

        static_img = 255 * ((static_img - static_img.min()) /
                            (static_img.max() - static_img.min()))
        moved_img = 255 * ((moved_img - moved_img.min()) /
                           (moved_img.max() - moved_img.min()))

        # Create the color images
        overlay = np.zeros(shape=static_img.shape + (3,), dtype=np.uint8)
        overlay[..., 0] = static_img
        overlay[..., 1] = moved_img
        mean, std = overlay[overlay > 0].mean(), overlay[overlay > 0].std()
        value_range = (mean - 0.5 * std, mean + 1.5 * std)
        return overlay, value_range

    def get_row_cols(self, num_slices):

        """
        Experimental helper function to get the number
        of rows and columns for the mosaic.

        num_slices: int
            The number of slices as obtained from the
            create_mosaic function.

        Returns
        -------
        rows, cols : int, int
            The number of rows and columns to be used in the grid for the
            mosaic.
        """
        rows, cols = 0, 0

        while True:
            if num_slices % 5 == 0:
                break
            num_slices += 1

        for i in range(5, num_slices):

            if num_slices % i == 0:
                rows = i
                cols = num_slices // i
                break

        return rows, cols

    def adjust_color_range(self, img):

        """
        Function to adjust the range of colors in numpy array
        to create the GIF (GIF standard only supports 256 colors).

        img : ndarray
            The numpy array containing the image data.

        Returns
        -------
        The numpy array with scaled down range of color values.
        """

        # Interpolating to range 0-14 for reducing the color values.
        img[..., 0] = np.interp(img[..., 0], (img[..., 0].min(),
                                              img[..., 0].max()), (0, 14))
        img[..., 1] = np.interp(img[..., 1], (img[..., 1].min(),
                                              img[..., 1].max()), (0, 14))
        img = np.round(img, 0).astype('uint8')

        # Interpolating the reduced range back to 0-255 for GIF.
        img[..., 0] = np.interp(img[..., 0], (img[..., 0].min(),
                                              img[..., 0].max()), (0, 255))
        img[..., 1] = np.interp(img[..., 1], (img[..., 1].min(),
                                              img[..., 1].max()), (0, 255))

        img = np.round(img, 0).astype('uint8')

        return img

    def create_mosaic(self, static_img, moved_img, affine, fname):
        """
        Function for creating the mosaic of the moved image. It
        currently only supports slices from the axial plane.

        static_img : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.

        moved_img : array, shape (S', R', C') or (R', C')
            the image to be used as "moving" during optimization.

        affine : the affine matrix to be used for transforming the moving
            image.

        fname: str, optional
            Filename to be used for saving the mosaic (default 'mosaic.png').

        Returns
        -------
            Saves a png file for the mosaic is saved on the disk.
        """

        overlay, value_range = self.process_image_data(static_img,
                                                       moved_img)

        print(overlay.shape)
        print(overlay[50,50,50,:])

        if fname is None:
            return overlay

        renderer = window.Renderer()
        renderer.background((0.5, 0.5, 0.5))

        slice_actor = actor.slicer(overlay, affine, value_range)
        renderer.projection('parallel')

        cnt = 0
        x, y, num_slices = slice_actor.shape[:3]

        rows, cols = self.get_row_cols(num_slices)
        border = 5

        for j in range(rows):
            for i in range(cols):
                slice_mosaic = slice_actor.copy()
                slice_mosaic.display(None, None, cnt)
                slice_mosaic.SetPosition((x + border) * i,
                                         0.5 * cols * (y + border) -
                                         (y + border) * j, 0)
                slice_mosaic.SetInterpolate(False)
                renderer.add(slice_mosaic)
                renderer.reset_camera()
                renderer.zoom(1.6)
                cnt += 1
                if cnt > num_slices:
                    break
            if cnt > num_slices:
                break

        renderer.reset_camera()
        renderer.zoom(1.6)

        window.record(renderer, out_path=fname,
                      size=(900, 600), reset_camera=False)

    def animate_overlap(self, static_img, moved_img,
                        slice_type, fname, affine):

        """
        Function for creating the animated GIF from the slices of the
        registered image.
        It uses the renderer object to control the dimensions of the
        created GIF and for correcting the orientation.

        Parameters
        ----------
        static_img : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.

        moved_img : array, shape (S', R', C') or (R', C')
            the image to be used as "moving" during optimization.

        slice_type : str (optional)
            the type of slice to be extracted:
            sagittal, coronal, axial.

        fname: str, optional
            Filename for saving the GIF (default 'animation.gif').

        Returns
        -------
            A GIF animation is saved on the disk.
        """

        overlay, value_range = self.process_image_data(static_img,
                                                       moved_img)
        x, y, z, _ = overlay.shape

        num_slices = 0

        if slice_type == 'sagittal':
            num_slices = x
            slice_type = 0
        elif slice_type == 'coronal':
            num_slices = y
            slice_type = 1
        elif slice_type == 'axial':
            num_slices = z
            slice_type = 2

        # Creating the renderer object and setting the background.
        renderer = window.renderer((0.5, 0.5, 0.5))
        slices = []

        for i in range(num_slices):
            temp_slice = overlay_slices(overlay[..., 0], overlay[..., 1],
                                        slice_type=slice_type,
                                        slice_index=i, ret_slice=True)
            temp_slice = temp_slice[..., None]
            temp_slice = np.round(temp_slice, 0).astype('uint8')
            temp_slice = np.rollaxis(temp_slice, 2, 4)
            slice_actor = actor.slicer(temp_slice, affine, value_range)
            renderer.add(slice_actor)
            renderer.reset_camera()
            renderer.zoom(1.6)
            snap = snapshot(renderer)
            snap1 = self.adjust_color_range(snap)
            slices.append(snap1)

        # Writing the GIF below
        write_gif(slices, fname, fps=10)

    @staticmethod
    def check_dimensions(static_img, moved_img):

        """
        Check the dimensions of the input images.

        Parameters
        ----------
        static_img : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.

        moved_img : array, shape (S', R', C') or (R', C')
            the image to be used as "moving" during optimization.

        """

        if len(static_img.shape) != len(moved_img.shape):
            raise ValueError('Dimension mismatch: The input images '
                             'must have same number of dimensions.')

    @staticmethod
    def check_slice_type(slice_type):
        if slice_type not in ['axial', 'sagittal', 'coronal']:
            raise ValueError('Unrecognized slice type {0}. Valid options are '
                             'axial, sagittal, coronal'.format(slice_type))

    def run(self, static_img_file, moving_img_file, affine_matrix_file,
            show_mosaic=False, anim_slice_type=None, out_dir='',
            mosaic_file='mosaic.png', animate_file='animation.gif'):
        """
        Parameters
        ----------
        static_img_file : string
            Path to the static image file.

        moving_img_file : string
            Path to the moving image file.

        affine_matrix_file : string
            The text file containing the affine matrix.

        show_mosaic : bool, optional
            If enabled, a mosaic of the all the slices from the
            axial plane is saved in an image (default False).

        anim_slice_type : str, optional
            A GIF animation showing the overlap of slices
            from static and moving images will be saved.
            The valid slice type(s): sagittal, coronal, axial, None(default).

        out_dir : string, optional
            Directory to save the results (default '').

        mosaic_file : string, optional
            Name of the file to save the mosaic (default 'mosaic.png').

        animate_file : string, optional
            Name of the GIF file for saving the animation
            (default animation.gif).

        """

        io_it = self.get_io_iterator()

        for static_img, mov_img, affine_matrix_file in io_it:

            # Load the data from the input files and store into objects.

            image = nib.load(static_img)
            static = image.get_data()

            image = nib.load(mov_img)
            moved_image = image.get_data()

            # Load the affine matrix from the input file and
            # store into numpy object.
            affine_matrix = load_affine_matrix(affine_matrix_file)

            # Do a sanity check on the number of dimensions.
            self.check_dimensions(static, moved_image)

            # Call the create_mosaic function if a valid option was provided.
            if show_mosaic:
                self.create_mosaic(static, moved_image, affine_matrix,
                                   mosaic_file)

            if anim_slice_type is not None:
                self.check_slice_type(anim_slice_type)
                self.animate_overlap(static, moved_image, anim_slice_type,
                                     animate_file, affine_matrix)

            if not show_mosaic and anim_slice_type is None:
                logging.info('No options given.')
