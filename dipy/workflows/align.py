from __future__ import division, print_function, absolute_import
import logging
import numpy as np
from dipy.align.reslice import reslice
from dipy.io.image import load_nifti, save_nifti
from dipy.workflows.workflow import Workflow
from dipy.align.streamlinear import slr_with_qbx
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import transform_streamlines


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
