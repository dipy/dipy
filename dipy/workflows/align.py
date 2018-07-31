from __future__ import division, print_function, absolute_import

import logging
import numpy as np
import nibabel as nib

from dipy.align.imaffine import AffineMap, transform_centers_of_mass, \
    MutualInformationMetric, AffineRegistration
from dipy.align.reslice import reslice
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D,
                                   AffineTransform3D)
from dipy.align.streamlinear import slr_with_qbx
from dipy.io.image import save_nifti, load_nifti, save_qa_metric
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import transform_streamlines
from dipy.workflows.workflow import Workflow

import numpy as np
import nibabel as nib

from dipy.align.reslice import reslice
from dipy.align.imaffine import AffineMap, transform_centers_of_mass, \
    MutualInformationMetric, AffineRegistration
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, \
    AffineTransform3D
from dipy.io.image import save_nifti, load_nifti, load_affine_matrix, \
    save_affine_matrix, save_quality_assur_metric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric


class UtilMethods(object):

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
        if metric not in ['mi', 'cc']:
            raise ValueError('Invalid similarity metric: Please provide'
                             ' a valid metric.')


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


class SlrWithQbxFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'slrwithqbx'

    def run(self, static_files, moving_files,
            x0='affine',
            rm_small_clusters=50,
            qbx_thr=[40, 30, 20, 15],
            num_threads=None,
            greater_than=50,
            less_than=250,
            nb_pts=20,
            progressive=True,
            out_dir='',
            out_moved='moved.trk',
            out_affine='affine.txt',
            out_stat_centroids='static_centroids.trk',
            out_moving_centroids='moving_centroids.trk',
            out_moved_centroids='moved_centroids.trk'):
        """ Streamline-based linear registration.

        For efficiency we apply the registration on cluster centroids and
        remove small clusters.

        Parameters
        ----------
        static_files : string
        moving_files : string
        x0 : string, optional
            rigid, similarity or affine transformation model (default affine)
        rm_small_clusters : int, optional
            Remove clusters that have less than `rm_small_clusters`
            (default 50)
        qbx_thr : variable int, optional
            Thresholds for QuickBundlesX (default [40, 30, 20, 15])
        num_threads : int, optional
            Number of threads. If None (default) then all available threads
            will be used. Only metrics using OpenMP will use this variable.
        greater_than : int, optional
            Keep streamlines that have length greater than
            this value (default 50)
        less_than : int, optional
            Keep streamlines have length less than this value (default 250)
        np_pts : int, optional
            Number of points for discretizing each streamline (default 20)
        progressive : boolean, optional
            (default True)
        out_dir : string, optional
            Output directory (default input file directory)
        out_moved : string, optional
            Filename of moved tractogram (default 'moved.trk')
        out_affine : string, optional
            Filename of affine for SLR transformation (default 'affine.txt')
        out_stat_centroids : string, optional
            Filename of static centroids (default 'static_centroids.trk')
        out_moving_centroids : string, optional
            Filename of moving centroids (default 'moving_centroids.trk')
        out_moved_centroids : string, optional
            Filename of moved centroids (default 'moved_centroids.trk')

        Notes
        -----
        The order of operations is the following. First short or long
        streamlines are removed. Second the tractogram or a random selection
        of the tractogram is clustered with QuickBundlesX. Then SLR
        [Garyfallidis15]_ is applied.

        References
        ----------
        .. [Garyfallidis15] Garyfallidis et al. "Robust and efficient linear
        registration of white-matter fascicles in the space of
        streamlines", NeuroImage, 117, 124--140, 2015

        .. [Garyfallidis14] Garyfallidis et al., "Direct native-space fiber
        bundle alignment for group comparisons", ISMRM, 2014.

        .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
        bundles using local and global streamline-based registration
        and clustering, Neuroimage, 2017.
        """
        io_it = self.get_io_iterator()

        logging.info("QuickBundlesX clustering is in use")
        logging.info('QBX thresholds {0}'.format(qbx_thr))

        for static_file, moving_file, out_moved_file, out_affine_file, \
                static_centroids_file, moving_centroids_file, \
                moved_centroids_file in io_it:

            logging.info('Loading static file {0}'.format(static_file))
            logging.info('Loading moving file {0}'.format(moving_file))

            static, static_header = load_trk(static_file)
            moving, moving_header = load_trk(moving_file)

            moved, affine, centroids_static, centroids_moving = \
                slr_with_qbx(
                    static, moving, x0, rm_small_clusters=rm_small_clusters,
                    greater_than=greater_than, less_than=less_than,
                    qbx_thr=qbx_thr)

            logging.info('Saving output file {0}'.format(out_moved_file))
            save_trk(out_moved_file, moved, affine=np.eye(4),
                     header=static_header)

            logging.info('Saving output file {0}'.format(out_affine_file))
            np.savetxt(out_affine_file, affine)

            logging.info('Saving output file {0}'
                         .format(static_centroids_file))
            save_trk(static_centroids_file, centroids_static, affine=np.eye(4),
                     header=static_header)

            logging.info('Saving output file {0}'
                         .format(moving_centroids_file))
            save_trk(moving_centroids_file, centroids_moving,
                     affine=np.eye(4),
                     header=static_header)

            centroids_moved = transform_streamlines(centroids_moving, affine)

            logging.info('Saving output file {0}'
                         .format(moved_centroids_file))
            save_trk(moved_centroids_file, centroids_moved, affine=np.eye(4),
                     header=static_header)


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
    registration will be slower but will improve the quality.

    This can be controlled by using the progressive flag (True by default).
    """

    def perform_transformation(self, static, static_grid2world, moving,
                               moving_grid2world,
                               affreg, params0, transform, affine):

        """ Function to apply the transformation.

        Parameters
        ----------
        static : 2D or 3D array
            the image to be used as reference during optimization.

        static_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the static
            image. The default is None, implying the transform is the
            identity.

        moving : 2D or 3D array
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

        transform : An instance of transform type.

        affine : Affine matrix to be used as starting affine
        """
        img_registration, \
            xopt, fopt = affreg.optimize(static, moving, transform, params0,
                                         static_grid2world,
                                         moving_grid2world,
                                         starting_affine=affine,
                                         ret_metric=True)
        transformed = img_registration.transform(moving)
        return transformed, img_registration.affine, xopt, fopt

    def center_of_mass(self, static, static_grid2world,
                       moving, moving_grid2world):

        """ Function for the center of mass based image registration.

        Parameters
        ----------
        static : 2D or 3D array
            the image to be used as reference during optimization.

        static_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the static
            image. The default is None, implying the transform is the
            identity.

        moving : 2D or 3D array
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

        """ Function for translation based registration.

        Parameters
        ----------
        static : 2D or 3D array
            the image to be used as reference during optimization.

        static_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the static
            image. The default is None, implying the transform is the
            identity.

        moving : 2D or 3D  array
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
        _, affine = self.center_of_mass(static, static_grid2world, moving,
                                        moving_grid2world)

        transform = TranslationTransform3D()
        return self.perform_transformation(static, static_grid2world,
                                           moving, moving_grid2world,
                                           affreg, params0, transform,
                                           affine)

    def rigid(self, static, static_grid2world, moving, moving_grid2world,
              affreg, params0, progressive):

        """ Function for rigid body based image registration.

        Parameters
        ----------
        static : 2D or 3D array
            the image to be used as reference during optimization.

        static_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the static
            image. The default is None, implying the transform is the
            identity.

        moving : 2D or 3D array
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
            _, affine, xopt, fopt = self.translate(static, static_grid2world,
                                                   moving, moving_grid2world,
                                                   affreg, params0)

        else:
            _, affine = self.center_of_mass(static, static_grid2world, moving,
                                            moving_grid2world)

        transform = RigidTransform3D()
        return self.perform_transformation(static, static_grid2world,
                                           moving, moving_grid2world,
                                           affreg, params0, transform,
                                           affine)

    def affine(self, static, static_grid2world, moving, moving_grid2world,
               affreg, params0, progressive):

        """ Function for full affine registration.

        Parameters
        ----------
        static : 2D or 3D array
            the image to be used as reference during optimization.

        static_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the static
            image. The default is None, implying the transform is the
            identity.

        moving : 2D or 3D array
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
            _, affine, xopt, fopt = self.rigid(static, static_grid2world,
                                               moving, moving_grid2world,
                                               affreg, params0, progressive)
        else:
            _, affine = self.center_of_mass(static, static_grid2world,
                                            moving, moving_grid2world)

        transform = AffineTransform3D()
        return self.perform_transformation(static, static_grid2world,
                                           moving, moving_grid2world,
                                           affreg, params0, transform,
                                           affine)

    @staticmethod
    def check_dimensions(static, moving):

        """
        Check the dimensions of the input images.

        Parameters
        ----------
        static : 2D or 3D array
            the image to be used as reference during optimization.

        moving: 2D or 3D array
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed
            to be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the
            'starting_affine' matrix
        """
        if len(static.shape) != len(moving.shape):
            raise ValueError('Dimension mismatch: The input images must '
                             'have same number of dimensions.')

        if len(static.shape) > 3 and len(moving.shape) > 3:
            raise ValueError('Dimension mismatch: One of the input should '
                             'be 2D or 3D dimensions.')

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
            Path to the static image file.

        moving_img_file : string
            Path to the moving image file.

        transform : string, optional
            com: center of mass, trans: translation, rigid: rigid body
             affine: full affine including translation, rotation, shearing and
             scaling (default 'affine').

        nbins : int, optional
            Number of bins to discretize the joint and marginal PDF
             (default '32').

        sampling_prop : int, optional
            Number ([0-100]) of voxels for calculating the PDF.
             'None' implies all voxels (default 'None').

        metric : string, optional
            Similarity metric for gathering mutual information
             (default 'mi' , Mutual Information metric).

        level_iters : variable int, optional
            The number of iterations at each scale of the scale space.
             `level_iters[0]` corresponds to the coarsest scale,
             `level_iters[-1]` the finest, where n is the length of the
              sequence. By default, a 3-level scale space with iterations
              sequence equal to [10000, 1000, 100] will be used.

        sigmas : variable floats, optional
            Custom smoothing parameter to build the scale space (one parameter
             for each scale). By default, the sequence of sigmas will be
             [3, 1, 0].

        factors : variable floats, optional
            Custom scale factors to build the scale space (one factor for each
             scale). By default, the sequence of factors will be [4, 2, 1].

        progressive : boolean, optional
            Enable/Disable the progressive registration (default 'True').

        save_metric : boolean, optional
            If true, quality assessment metric are saved in
            'quality_metric.txt' (default 'False').

        out_dir : string, optional
            Directory to save the transformed image and the affine matrix
             (default '').

        out_moved : string, optional
            Name for the saved transformed image
             (default 'moved.nii.gz').

        out_affine : string, optional
            Name for the saved affine matrix
             (default 'affine.txt').

        out_quality : string, optional
            Name of the file containing the saved quality
             metric (default 'quality_metric.txt').
        """

        io_it = self.get_io_iterator()
        transform = transform.lower()

        for static_img, mov_img, moved_file, affine_matrix_file, \
                qual_val_file in io_it:

            # Load the data from the input files and store into objects.
            image = nib.load(static_img)
            static = np.array(image.get_data())
            static_grid2world = image.affine

            image = nib.load(mov_img)
            moving = np.array(image.get_data())
            moving_grid2world = image.affine

            util = UtilMethods()

            if transform == 'com':
                moved_image, affine = self.center_of_mass(static,
                                                          static_grid2world,
                                                          moving,
                                                          moving_grid2world)
            else:

                params0 = None
                if metric != 'mi':
                    raise ValueError("Invalid similarity metric: Please"
                                     " provide a valid metric.")
                metric = MutualInformationMetric(nbins, sampling_prop)

                """
                Instantiating the registration class with the configurations.
                """

                affreg = AffineRegistration(metric=metric,
                                            level_iters=level_iters,
                                            sigmas=sigmas,
                                            factors=factors)

                if transform == 'trans':
                    moved_image, affine, \
                        xopt, fopt = self.translate(static,
                                                    static_grid2world,
                                                    moving,
                                                    moving_grid2world,
                                                    affreg,
                                                    params0)

                elif transform == 'rigid':
                    moved_image, affine, \
                        xopt, fopt = self.rigid(static,
                                                static_grid2world,
                                                moving,
                                                moving_grid2world,
                                                affreg,
                                                params0,
                                                progressive)

                elif transform == 'affine':
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
                logging.info("Optimal parameters: {0}".format(str(xopt)))
                logging.info("Similarity metric: {0}".format(str(fopt)))

                if save_metric:
                    save_qa_metric(qual_val_file, xopt, fopt)

            save_nifti(moved_file, moved_image, static_grid2world)
            np.savetxt(affine_matrix_file, affine)


class ApplyAffineFlow(Workflow):

    def run(self, static_image_file, moving_image_files, affine_matrix_file,
            out_dir='', out_file='transformed.nii.gz'):

        """
        Parameters
        ----------
        static_image_file : string
            Path of the static image file.

        moving_image_files : string
            Path of the moving image(s). It can be a single image or a
            folder containing multiple images.

        affine_matrix_file : string
            The text file containing the affine matrix for transformation.

        out_dir : string, optional
            Directory to save the transformed files (default '').

        out_file : string, optional
            Name of the transformed file (default 'transformed.nii.gz').
             It is recommended to use the flag --mix-names to
              prevent the output files from being overwritten.

        """
        io = self.get_io_iterator()

        for static_image_file, moving_image_file, affine_matrix_file, \
                out_file in io:

            # Loading the image data from the input files into object.
            static_image, static_grid2world = load_nifti(static_image_file)

            moving_image, moving_grid2world = load_nifti(moving_image_file)

            # Doing a sanity check for validating the dimensions of the input
            # images.
            ImageRegistrationFlow.check_dimensions(static_image, moving_image)

            # Loading the affine matrix.
            affine_matrix = np.loadtxt(affine_matrix_file)

            # Setting up the affine transformation object.
            img_transformation = AffineMap(
                affine=affine_matrix,
                domain_grid_shape=static_image.shape,
                domain_grid2world=static_grid2world,
                codomain_grid_shape=moving_image.shape,
                codomain_grid2world=moving_grid2world)

            # Transforming the image/
            transformed = img_transformation.transform(moving_image)

            save_nifti(out_file, transformed, affine=static_grid2world)


class SynRegistrationFlow(Workflow):

    def run(self, static_image_file, moving_image_file, affine_matrix_file,
            level_iters=[10, 10, 5], metric="cc", out_dir='',
            out_warped='warped_moved.nii.gz'):

        """
        Parameters
        ----------
        static_image_file : string
            Path of the static image file.

        moving_image_file : string
            Path to the moving image file.

        affine_matrix_file : string
            The text file containing pre alignment information or the
            affine matrix.

        level_iters : variable int, optional
            The number of iterations at each level of the gaussian pyramid.
            By default, a 3-level scale space with iterations
            sequence equal to [10, 10, 5] will be used. The 0-th
            level corresponds to the finest resolution.

        metric : string, optional
            The metric to be used (Default cc, 'Cross Correlation metric').

        out_dir : string, optional
            Directory to save the transformed files (default '').

        out_warped : string, optional
            Name of the warped file. If no name is given then a
            suffix 'transformed' will be appended to the name of the
            original input file (default 'warped_moved.nii.gz').

        """

        io = self.get_io_iterator()
        util = UtilMethods()
        util.check_metric(metric)

        for static_file, moving_file, in_affine, \
                warped_file in io:

            print(static_file, moving_file, in_affine, warped_file)

            # Loading the image data from the input files into object.
            static_img_data = nib.load(static_file)
            static_image = static_img_data.get_data()
            static_grid2world = static_img_data.affine

            moving_img_data = nib.load(moving_file)
            moving_image = moving_img_data.get_data()
            moving_grid2world = moving_img_data.affine

            # Sanity check for the input image dimensions.
            util.check_dimensions(static_image, moving_image)

            # Loading the affine matrix.
            affine_matrix = load_affine_matrix(in_affine)

            metric = CCMetric(3)
            sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

            mapping = sdr.optimize(static_image, moving_image,
                                   static_grid2world, moving_grid2world,
                                   affine_matrix)

            warped_moving = mapping.transform(moving_image)

            # Saving the warped moving file and the alignment matrix.
            save_nifti(warped_file, warped_moving, static_grid2world)
