from __future__ import division, print_function, absolute_import
import logging
from dipy.align.reslice import reslice
from dipy.workflows.workflow import Workflow

import numpy as np
import nibabel as nib
from dipy.align.imaffine import (transform_centers_of_mass, AffineMap,
                                 MutualInformationMetric, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D,
                                   AffineTransform3D)
from dipy.io.image import load_nifti, save_nifti, save_affine_matrix, \
    save_quality_assur_metric

from dipy.viz.regtools import overlay_images
from dipy.segment.mask import median_otsu

from dipy.viz import window, actor, ui
from dipy.viz.window import snapshot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ResliceFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'reslice'

    def run(self, input_files, new_vox_size, order=1, mode='constant', cval=0,
            num_processes=1, out_dir='', out_resliced='resliced.nii.gz'):
        """Reslice data with new voxel resolution defined by ``new_vox_sz``

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        new_vox_size : variable float
            new voxel size
        order : int, optional
            order of interpolation, from 0 to 5, for resampling/reslicing,
            0 nearest interpolation, 1 trilinear etc.. if you don't want any
            smoothing 0 is the option you need (default 1)
        mode : string, optional
            Points outside the boundaries of the input are filled according
            to the given mode 'constant', 'nearest', 'reflect' or 'wrap'
            (default 'constant')
        cval : float, optional
            Value used for points outside the boundaries of the input if
            mode='constant' (default 0)
        num_processes : int, optional
            Split the calculation to a pool of children processes. This only
            applies to 4D `data` arrays. If a positive integer then it defines
            the size of the multiprocessing pool that will be used. If 0, then
            the size of the pool will equal the number of cores available.
            (default 1)
        out_dir : string, optional
            Output directory (default input file directory)
        out_resliced : string, optional
            Name of the resliced dataset to be saved
            (default 'resliced.nii.gz')
        """

        io_it = self.get_io_iterator()

        for inputfile, outpfile in io_it:
            data, affine, vox_sz = load_nifti(inputfile, return_voxsize=True)
            logging.info('Processing {0}'.format(inputfile))
            new_data, new_affine = reslice(data, affine, vox_sz, new_vox_size,
                                           order, mode=mode, cval=cval,
                                           num_processes=num_processes)
            save_nifti(outpfile, new_data, new_affine)
            logging.info('Resliced file save in {0}'.format(outpfile))


class ImageRegistrationFlow(Workflow):
    """
    The registration workflow is organized as a collection of different
    functions. The user can intend to use only one type of registration
    (such as center of mass or rigid body registration only).

    Alternatively, a registration can be done in a progressive manner.
    For example, using affine registration with progressive set to 'True'
    will involve center of mass, translation, rigid body and full affine
    registration. Whereas, when progressive is False the registration will
    include only center of mass and affine registration. The progressive
    registration will be slower but will improve the quality of the results.

    This can be controlled by using the progressive flag (True by default).
    """

    def test_visual(self, static_img, static_grid2world, moved_img,
                               moving_grid2world):

        # Normalize the input images to [0,255]
        static_img = 255 * ((static_img - static_img.min()) / (static_img.max() - static_img.min()))
        moved_img = 255 * ((moved_img - moved_img.min()) / (moved_img.max() - moved_img.min()))

        # Create the color images
        overlay = np.zeros(shape=(static_img.shape) + (3,), dtype=np.uint8)

        # Copy the normalized intensities into the appropriate channels of the
        # color images
        overlay[..., 0] = static_img
        overlay[..., 1] = moved_img

        data = overlay
        affine = moving_grid2world

        renderer = window.Renderer()
        renderer.background((0.5, 0.5, 0.5))

        mean, std = data[data > 0].mean(), data[data > 0].std()
        value_range = (mean - 0.5 * std, mean + 1.5 * std)

        lut = actor.colormap_lookup_table(scale_range=(0, 255),
                                          hue_range=(0.4, 1.),
                                          saturation_range=(1, 1.),
                                          value_range=(0., 1.))

        slice_actor = actor.slicer(data, affine, value_range, lookup_colormap=lut)
        #slice_actor2 = slice_actor.copy()
        renderer.reset_camera()
        renderer.zoom(1.4)
        # for i in range(slice_actor.shape[2]):
        #     slice_actor2.display(None, None, i)
        #     renderer.add(slice_actor2)
        #     window.show(renderer, size=(600, 600), reset_camera=False)
        #     print(i)

        window.record(renderer, out_path='slices.gif', size=(600, 600),
                      reset_camera=False)
        renderer.projection('parallel')

        result_position = ui.TextBlock2D(text='')
        result_value = ui.TextBlock2D(text='')

        result_position.message = ''
        result_value.message = ''

        show_m_mosaic = window.ShowManager(renderer, size=(1200, 900))
        show_m_mosaic.initialize()
        cnt = 0
        X, Y, Z = slice_actor.shape

        rows = 5
        cols = 15
        border = 10

        for j in range(rows):
            for i in range(cols):
                slice_mosaic = slice_actor.copy()
                slice_mosaic.display(None, None, cnt)
                slice_mosaic.SetPosition((X + border) * i,
                                         0.5 * cols * (Y + border) - (Y + border) * j,
                                         0)
                slice_mosaic.SetInterpolate(False)
                renderer.add(slice_mosaic)
                cnt += 1
                if cnt > Z:
                    break
            if cnt > Z:
                break

        renderer.reset_camera()
        renderer.zoom(1.6)

        #show_m_mosaic.ren.add(panel_picking)
        #show_m_mosaic.start()

        window.record(renderer, out_path='mosaic.png', size=(900, 600),
                      reset_camera=False)

    def create_overlaid_images(self, static_img, static_grid2world, moved_img,
                               moving_grid2world):

        identity = np.eye(4)
        affine_map = AffineMap(identity,
                               static_img.shape, static_grid2world,
                               moved_img.shape, moving_grid2world)
        resampled = affine_map.transform(moved_img)
        b0_mask, mask = median_otsu(static_img, 4, 4)
        t1_mask, mask = median_otsu(resampled, 4, 4)

        static = b0_mask[:, :, 60]
        moving = t1_mask[:, :, 60]
        overlay_images(static, moving, 'static', 'registered', 'moving', fname='myimage.png')

    def visualsss(self, static_img, static_grid2world, moved_img,
                               moving_grid2world):

        # Normalize the input images to [0,255]
        static_img = 255 * ((static_img - static_img.min()) / (static_img.max() - static_img.min()))
        moved_img = 255 * ((moved_img - moved_img.min()) / (moved_img.max() - moved_img.min()))

        # Create the color images
        overlay = np.zeros(shape=(static_img.shape) + (3,), dtype=np.uint8)

        # Copy the normalized intensities into the appropriate channels of the
        # color images
        overlay[..., 0] = static_img
        overlay[..., 1] = moved_img

        fig, ax = plt.subplots()
        ln, = plt.plot([], 'ro', animated=True)

        mean, std = overlay[overlay > 0].mean(), overlay[overlay > 0].std()
        value_range = (mean - 0.5 * std, mean + 1.5 * std)

        my_color_map = actor.colormap_lookup_table(scale_range=(0, 255),
                                          hue_range=(0.7, 1.),
                                          saturation_range=(1, 1.),
                                          value_range=(0.3, 1.))

        slice_actor = actor.slicer(overlay, static_grid2world, value_range)
        print(slice_actor.shape[:3])
        rend = window.renderer()
        rend.background((0.5,0.5,0.5))
        rend.add(slice_actor)
        myarr = snapshot(rend)
        slice_ext =  slice_actor.copy()



        def update(frame):
            # plt.imshow(overlay[:, :, frame, 0])
            # plt.imshow(overlay[:, :, frame, 1])
            plt.imshow(myarr[:,:,frame])
            #plt.imshow(slice_ext.display(None, None, frame))
            return ln,

        ani = FuncAnimation(fig, update, frames=range(3),
                            blit=True)

        ani.save('dynamic_images.html')


    def center_of_mass(self, static, static_grid2world,
                       moving, moving_grid2world):

        """ Function for the center of mass based image
        registration.

        Parameters
        ----------
        static : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.

        static_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the static
            image. The default is None, implying the transform is the
            identity.

        moving : array, shape (S', R', C') or (R', C')
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed
            to be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the
            'starting_affine' matrix

        moving_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the moving
            image. The default is None, implying the transform is the
            identity.

        """

        img_registration = transform_centers_of_mass(static,
                                                     static_grid2world,
                                                     moving,
                                                     moving_grid2world)

        transformed = img_registration.transform(moving)
        return transformed, img_registration.affine

    def translate(self, static, static_grid2world, moving,
                  moving_grid2world, affreg, params0):

        """ Function for translating the image.

        Parameters
        ----------
        static : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.

        static_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the static
            image. The default is None, implying the transform is the
            identity.

        moving : array, shape (S', R', C') or (R', C')
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed
            to be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the
            'starting_affine' matrix

        moving_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the moving
            image. The default is None, implying the transform is the
            identity.

        affreg : An object of the image registration class.

        params0 : array, shape (n,)
            parameters from which to start the optimization. If None, the
            optimization will start at the identity transform. n is the
            number of parameters of the specified transformation.

        """
        moved, affine = self.center_of_mass(static, static_grid2world,
                                            moving, moving_grid2world)

        transform = TranslationTransform3D()
        starting_affine = affine

        img_registration, \
            xopt, fopt = affreg.optimize(static, moving, transform,
                                         params0, static_grid2world,
                                         moving_grid2world,
                                         starting_affine=starting_affine,
                                         ret_metric=True)

        transformed = img_registration.transform(moving)
        return transformed, img_registration.affine, xopt, fopt

    def rigid(self, static, static_grid2world, moving, moving_grid2world,
              affreg, params0, progressive):

        """ Function for rigid image registration.

        Parameters
        ----------
        static : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.

        static_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the static
            image. The default is None, implying the transform is the
            identity.

        moving : array, shape (S', R', C') or (R', C')
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed
            to be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the
            'starting_affine' matrix

        moving_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the moving
            image. The default is None, implying the transform is the
            identity.

        affreg : An object of the image registration class.

        params0 : array, shape (n,)
            parameters from which to start the optimization. If None, the
            optimization will start at the identity transform. n is the
            number of parameters of the specified transformation.

        progressive : boolean
            Flag to enable or disable the progressive registration. (defa
            ult True)

        """

        if progressive:
            moved, affine, xopt, fopt = self.translate(static,
                                                       static_grid2world,
                                                       moving,
                                                       moving_grid2world,
                                                       affreg, params0)

        else:
            moved, affine = self.center_of_mass(static, static_grid2world,
                                                moving, moving_grid2world)

        transform = RigidTransform3D()
        starting_affine = affine

        img_registration, \
            xopt, fopt = affreg.optimize(static, moving, transform,
                                         params0, static_grid2world,
                                         moving_grid2world,
                                         starting_affine=starting_affine,
                                         ret_metric=True)

        transformed = img_registration.transform(moving)
        return transformed, img_registration.affine, xopt, fopt

    def affine(self, static, static_grid2world, moving, moving_grid2world,
               affreg, params0, progressive):

        """ Function for full affine registration.

        Parameters
        ----------
        static : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.

        static_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the static
            image. The default is None, implying the transform is the
            identity.

        moving : array, shape (S', R', C') or (R', C')
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed
            to be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the
            'starting_affine' matrix

        moving_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the moving
            image. The default is None, implying the transform is the
            identity.

        affreg : An object of the image registration class.

        params0 : array, shape (n,)
            parameters from which to start the optimization. If None, the
            optimization will start at the identity transform. n is the
            number of parameters of the specified transformation.

        progressive : boolean
            Flag to enable or disable the progressive registration. (defa
            ult True)

        """
        if progressive:
            moved, affine, xopt, fopt = self.rigid(static, static_grid2world,
                                                   moving, moving_grid2world,
                                                   affreg, params0,
                                                   progressive)

        else:
            moved, affine = self.center_of_mass(static, static_grid2world,
                                                moving, moving_grid2world)

        transform = AffineTransform3D()
        starting_affine = affine

        img_registration, \
            xopt, fopt = affreg.optimize(static, moving, transform,
                                         params0, static_grid2world,
                                         moving_grid2world,
                                         starting_affine=starting_affine,
                                         ret_metric=True)

        transformed = img_registration.transform(moving)
        return transformed, img_registration.affine, xopt, fopt

    @staticmethod
    def check_dimensions(static, moving):

        """
        Check the dimensions of the input images.

        Parameters
        ----------
        static : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.
        moving: array, shape (S', R', C') or (R', C')
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed
            to be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the
            'starting_affine' matrix
        """
        if len(static.shape) != len(moving.shape):
            raise ValueError('Dimension mismatch: The'
                             ' input images must have same number of '
                             'dimensions.')

    @staticmethod
    def check_metric(metric):
        """
        Check the input metric type.

        Parameters
        ----------
        metric: string
            The similarity metric.
            (default 'MutualInformation' metric)

        """
        if metric != 'mi':
            raise ValueError('Invalid similarity metric: Please provide'
                             ' a valid metric.')

    def run(self, static_img_file, moving_img_file, transform='affine',
            nbins=32, sampling_prop=None, metric='mi',
            level_iters=[10000, 1000, 100], sigmas=[3.0, 1.0, 0.0],
            factors=[4, 2, 1], progressive=True, save_metric=False,
            out_dir='', out_moved='moved.nii.gz', out_affine='affine.txt',
            out_quality='quality_metric.txt'):

        """
        Parameters
        ----------
        static_img_file : string
            Path to the reference image.

        moving_img_file : string
            Path to the moving image file.

        transform : string, optional
             com : center of mass

            'trans' translation

            'rigid' rigid body

            'affine' full affine including translation, rotation, shearing and
             scaling (default 'affine')

        nbins : int, optional
            The number of bins to discretize the joint and marginal PDF. (def
            ault '32')

        sampling_prop : int, optional
            Number ([0-100]) of voxels to be used for calculati
            ng the PDF. 'None' implies all voxels. (default 'None')

        metric : string, optional
            The similarity metric to be used for gathering mutual information
            . (default 'Mutual Information metric')

        level_iters : variable int, optional
            the number of iterations at each scale of the scale space.
            `level_iters[0]` corresponds to the coarsest scale,
            `level_iters[-1]` the finest, where n is the length of the
            sequence. By default, a 3-level scale space with iterations
            sequence equal to [10000, 1000, 100] will be used.

        sigmas : variable floats, optional
            custom smoothing parameter to build the scale space (one parameter
            for each scale). By default, the sequence of sigmas will be
            [3, 1, 0].

        factors : variable floats, optional
            custom scale factors to build the scale space (one factor for each
            scale). By default, the sequence of factors will be [4, 2, 1].

        progressive : boolean, optional
            Flag for enabling/disabling the progressive registration.
            (default 'True')


        save_metric : boolean, optional
            If true, the metric values are
            saved in a file called 'quality_metric.txt'
            (default 'False')

            By default, the similarity measure
            values such as the distance and the
            metric of optimal parameters is only
            displayed but not saved.

        out_dir : string, optional
            Directory to save the transformed image and the affine matrix.
            (default '')

        out_moved : string, optional
            Name for the saved transformed image.
            (default 'moved.nii.gz')

        out_affine : string, optional
            Name for the saved affine matrix.
            (default 'affine.txt')

        out_quality : string, optional
            Name of the file containing the saved quality
            metrices (default 'quality_metric.txt')

        """

        """
        setting up the io iterator to gobble the input and output paths
        """
        io_it = self.get_io_iterator()

        for static_img, mov_img, moved_file, affine_matrix_file, \
                qual_val_file in io_it:

            """
            Load the data from the input files and store into objects.
            """
            image = nib.load(static_img)
            static = np.array(image.get_data())
            static_grid2world = image.affine

            image = nib.load(mov_img)
            moving = np.array(image.get_data())
            moving_grid2world = image.affine

            self.check_dimensions(static, moving)

            if transform.lower() == 'com':
                moved_image, affine = self.center_of_mass(static,
                                                          static_grid2world,
                                                          moving,
                                                          moving_grid2world)
                self.visualsss(static, static_grid2world,
                                 moved_image, moving_grid2world)
                # self.test_visual(static, static_grid2world,
                #                  moved_image, moving_grid2world)
            else:

                params0 = None
                self.check_metric(metric)
                metric = MutualInformationMetric(nbins, sampling_prop)

                """
                Instantiating the registration class with the configurations.
                """

                affreg = AffineRegistration(metric=metric,
                                            level_iters=level_iters,
                                            sigmas=sigmas,
                                            factors=factors)

                if transform.lower() == 'trans':
                    moved_image, affine, \
                        xopt, fopt = self.translate(static,
                                                    static_grid2world,
                                                    moving,
                                                    moving_grid2world,
                                                    affreg,
                                                    params0)

                elif transform.lower() == 'rigid':
                    moved_image, affine, \
                        xopt, fopt = self.rigid(static,
                                                static_grid2world,
                                                moving,
                                                moving_grid2world,
                                                affreg,
                                                params0,
                                                progressive)

                elif transform.lower() == 'affine':
                    moved_image, affine, \
                        xopt, fopt = self.affine(static,
                                                 static_grid2world,
                                                 moving,
                                                 moving_grid2world,
                                                 affreg,
                                                 params0,
                                                 progressive)
                else:
                    raise ValueError('Invalid transformation:'
                                     ' Please see program\'s help'
                                     ' for allowed values of'
                                     ' transformation.')

                """
                Saving the moved image file and the affine matrix.
                """

                self.visualsss(static, static_grid2world,
                                            moved_image, moving_grid2world)
                logging.info("Similarity metric:"+str(xopt))
                logging.info("Distance measure:"+str(fopt))

                if save_metric:
                    save_quality_assur_metric(qual_val_file, xopt, fopt)

            save_nifti(moved_file, moved_image, static_grid2world)
            save_affine_matrix(affine_matrix_file, affine)
