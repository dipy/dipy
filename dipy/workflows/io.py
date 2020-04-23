import os
import sys

import numpy as np
import logging
import importlib
from inspect import getmembers, isfunction
from dipy.io.image import load_nifti, save_nifti
from dipy.io.peaks import save_peaks, peaks_to_niftis, arrays_to_peaks
from dipy.workflows.workflow import Workflow


class IoInfoFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'io_info'

    def run(self, input_files,
            b0_threshold=50, bvecs_tol=0.01, bshell_thr=100):
        """ Provides useful information about different files used in
        medical imaging. Any number of input files can be provided. The
        program identifies the type of file by its extension.

        Parameters
        ----------
        input_files : variable string
            Any number of Nifti1, bvals or bvecs files.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Threshold used to check that norm(bvec) = 1 +/- bvecs_tol
            b-vectors are unit vectors.
        bshell_thr : float, optional
            Threshold for distinguishing b-values in different shells.
        """
        np.set_printoptions(3, suppress=True)

        io_it = self.get_io_iterator()

        for input_path in io_it:
            mult_ = len(input_path)
            logging.info('-----------' + mult_*'-')
            logging.info('Looking at {0}'.format(input_path))
            logging.info('-----------' + mult_*'-')

            ipath_lower = input_path.lower()

            if ipath_lower.endswith('.nii') or ipath_lower.endswith('.nii.gz'):

                data, affine, img, vox_sz, affcodes = load_nifti(
                    input_path,
                    return_img=True,
                    return_voxsize=True,
                    return_coords=True)
                logging.info('Data size {0}'.format(data.shape))
                logging.info('Data type {0}'.format(data.dtype))

                if data.ndim == 3:
                    logging.info('Data min {0} max {1} avg {2}'
                                 .format(data.min(), data.max(), data.mean()))
                    logging.info('2nd percentile {0} 98th percentile {1}'
                                 .format(np.percentile(data, 2),
                                         np.percentile(data, 98)))
                if data.ndim == 4:
                    logging.info('Data min {0} max {1} avg {2} of vol 0'
                                 .format(data[..., 0].min(),
                                         data[..., 0].max(),
                                         data[..., 0].mean()))
                    msg = '2nd percentile {0} 98th percentile {1} of vol 0'
                    logging.info(msg
                                 .format(np.percentile(data[..., 0], 2),
                                         np.percentile(data[..., 0], 98)))
                logging.info('Native coordinate system {0}'
                             .format(''.join(affcodes)))
                logging.info('Affine Native to RAS matrix \n{0}'.format(affine))
                logging.info('Voxel size {0}'.format(np.array(vox_sz)))
                if np.sum(np.abs(np.diff(vox_sz))) > 0.1:
                    msg = \
                        'Voxel size is not isotropic. Please reslice.\n'
                    logging.warning(msg)

            if os.path.basename(input_path).lower().find('bval') > -1:
                bvals = np.loadtxt(input_path)
                logging.info('b-values \n{0}'.format(bvals))
                logging.info('Total number of b-values {}'.format(len(bvals)))
                shells = np.sum(np.diff(np.sort(bvals)) > bshell_thr)
                logging.info('Number of gradient shells {0}'.format(shells))
                logging.info('Number of b0s {0} (b0_thr {1})\n'
                             .format(np.sum(bvals <= b0_threshold),
                                     b0_threshold))

            if os.path.basename(input_path).lower().find('bvec') > -1:

                bvecs = np.loadtxt(input_path)
                logging.info('Bvectors shape on disk is {0}'
                             .format(bvecs.shape))
                rows, cols = bvecs.shape
                if rows < cols:
                    bvecs = bvecs.T
                logging.info('Bvectors are \n{0}'.format(bvecs))
                norms = np.array([np.linalg.norm(bvec) for bvec in bvecs])
                res = np.where(
                        (norms <= 1 + bvecs_tol) & (norms >= 1 - bvecs_tol))
                ncl1 = np.sum(norms < 1 - bvecs_tol)
                logging.info('Total number of unit bvectors {0}'
                             .format(len(res[0])))
                logging.info('Total number of non-unit bvectors {0}\n'
                             .format(ncl1))

        np.set_printoptions()


class FetchFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'fetch'

    @staticmethod
    def get_fetcher_datanames():
        """Gets available dataset and function names.

        Returns
        -------
        available_data: dict
            Available dataset and function names.

        """

        fetcher_module = FetchFlow.load_module('dipy.data.fetcher')

        available_data = dict([(name.replace('fetch_', ''), func)
                               for name, func in getmembers(fetcher_module,
                                                            isfunction)
                               if name.lower().startswith("fetch_") and
                               func is not fetcher_module.fetch_data
                               if name.lower() not in ['fetch_hbn',
                                                       'fetch_hcp']])

        return available_data

    @staticmethod
    def load_module(module_path):
        """Load / reload an external module.

        Parameters
        ----------
        module_path: string
            the path to the module relative to the main script

        Returns
        -------
        module: module object

        """
        if module_path in sys.modules:
            return importlib.reload(sys.modules[module_path])
        else:
            return importlib.import_module(module_path)

    def run(self, data_names, out_dir=''):
        """Download files to folder and check their md5 checksums.

        To see all available datasets, please type "list" in data_names.

        Parameters
        ----------
        data_names : variable string
            Any number of Nifti1, bvals or bvecs files.
        out_dir : string, optional
            Output directory. (default current directory)

        """
        if out_dir:
            dipy_home = os.environ.get('DIPY_HOME', None)
            os.environ['DIPY_HOME'] = out_dir

        available_data = FetchFlow.get_fetcher_datanames()

        data_names = [name.lower() for name in data_names]

        if 'all' in data_names:
            for name, fetcher_function in available_data.items():
                logging.info('------------------------------------------')
                logging.info('Fetching at {0}'.format(name))
                logging.info('------------------------------------------')
                fetcher_function()

        elif 'list' in data_names:
            logging.info('Please, select between the following data names:'
                         ' {0}'.format(', '.join(available_data.keys())))

        else:
            skipped_names = []
            for data_name in data_names:
                if data_name not in available_data.keys():
                    skipped_names.append(data_name)
                    continue

                logging.info('------------------------------------------')
                logging.info('Fetching at {0}'.format(data_name))
                logging.info('------------------------------------------')
                available_data[data_name]()

            nb_success = len(data_names) - len(skipped_names)
            print('\n')
            logging.info('Fetched {0} / {1} Files '.format(nb_success,
                                                           len(data_names)))
            if skipped_names:
                logging.warn('Skipped data name(s):'
                             ' {0}'.format(' '.join(skipped_names)))
                logging.warn('Please, select between the following data'
                             ' names: {0}'.format(
                                 ', '.join(available_data.keys())))

        if out_dir:
            if dipy_home:
                os.environ['DIPY_HOME'] = dipy_home
            else:
                os.environ.pop('DIPY_HOME', None)

            # We load the module again so that if we run another one of these in
            # the same process, we don't have the env variable pointing to the
            # wrong place
            self.load_module('dipy.data.fetcher')


<<<<<<< HEAD
class SplitFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'split'

    def run(self, input_files, vol_idx=0, out_dir='',
            out_split='split.nii.gz'):
        """ Splits the input 4D file and extracts the required 3D volume.

        Parameters
        ----------
        input_files : variable string
            Any number of Nifti1 files
        vol_idx : int, optional
        out_dir : string, optional
            Output directory. (default current directory)
        out_split : string, optional
            Name of the resulting split volume

        """
        io_it = self.get_io_iterator()
        for fpath, osplit in io_it:
            logging.info('Splitting {0}'.format(fpath))
            data, affine, image = load_nifti(fpath, return_img=True)

            if vol_idx == 0:
                logging.info('Splitting and extracting 1st b0')

            split_vol = data[..., vol_idx]
            save_nifti(osplit, split_vol, affine, image.header)

            logging.info('Split volume saved as {0}'.format(osplit))
=======
class NiftisToPamFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'niftis_to_pam'

    def run(self, peaks_dir_files, peaks_values_files, peaks_indices_files,
            shm_files, gfa_files, out_dir='', out_pam="peaks.pam5"):
        """Convert pam5 files to mutiple nifti files.

        Parameters
        ----------
        peaks_dir_files : string
            Path to the input peaks directions volume. This path may contain
            wildcards to process multiple inputs at once.
        peaks_values_files : string
            Path to the input peaks values volume. This path may contain
            wildcards to process multiple inputs at once.
        peaks_indices_files : string
            Path to the input peaks indices volume. This path may contain
            wildcards to process multiple inputs at once.
        shm_files : string
            Path to the input spherical harmonics volume. This path may
            contain wildcards to process multiple inputs at once.
        gfa_files : string
            Path to the input generalized FA volume. This path may contain
            wildcards to process multiple inputs at once.
        out_dir : string, optional
            Output directory (default input file directory)
        out_pam : string, optional
            Name of the peaks volume to be saved (default 'peaks.pam5')

        """
        io_it = self.get_io_iterator()

        msg = 'pam5 files saved in '
        msg += out_dir or 'current directory'
        for fpeak_dirs, fpeak_values, fpeak_indices, fshm, fgfa, opam in io_it:
            logging.info('Converting nifti files to pam5')
            peak_dirs, affine = load_nifti(fpeak_dirs)
            peak_values, _ = load_nifti(fpeak_values)
            peak_indices, _ = load_nifti(fpeak_indices)
            shm, _ = load_nifti(fshm)
            gfa, _ = load_nifti(fgfa)

            arrays_to_peaks(affine=affine, peak_dirs=peak_dirs,
                            peak_values=peak_values, peak_indices=peak_indices,
                            pam_file=opam, gfa=gfa, shm_coeff=shm)
            logging.info(msg)


class PamToNiftisFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'pam_to_niftis'

    def run(self, pam_files, out_dir='', out_peaks_dir='peaks_dirs.nii.gz',
            out_peaks_values='peaks_values.nii.gz', out_gfa='gfa.nii.gz',
            out_peaks_indices='peaks_indices.nii.gz', out_shm='shm.nii.gz'):
        """Convert pam5 files to mutiple nifti files.

        Parameters
        ----------
        pam_files : string
            Path to the input peaks volumes. This path may contain wildcards to
            process multiple inputs at once.
        out_dir : string, optional
            Output directory (default input file directory)
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved
            (default 'peaks_dirs.nii.gz')
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved
            (default 'peaks_values.nii.gz')
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved
            (default 'peaks_indices.nii.gz')
        out_gfa : string, optional
            Name of the generalized FA volume to be saved (default 'gfa.nii.gz')
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved
            (default 'shm.nii.gz')

        """
        io_it = self.get_io_iterator()

        msg = 'Nifti files saved in '
        msg += out_dir or 'current directory'
        for pam, opeak_dirs, opeak_values, ogfa, opeak_indices, oshm in io_it:
            logging.info('Converting %s file to niftis...', pam)
            peaks_to_niftis(pam, fname_shm=oshm, fname_gfa=ogfa,
                            fname_dirs=opeak_dirs, fname_values=opeak_values,
                            fname_indices=opeak_indices)
            logging.info(msg)
>>>>>>> add converters
