from __future__ import division, print_function, absolute_import

import os
import sys
import numpy as np
import logging
import importlib
from inspect import getmembers, isfunction, getfullargspec
from dipy.io.image import load_nifti
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
            (default 50)
        bvecs_tol : float, optional
            Threshold used to check that norm(bvec) = 1 +/- bvecs_tol
            b-vectors are unit vectors (default 0.01)
        bshell_thr : float, optional
            Threshold for distinguishing b-values in different shells
            (default 100)
        """
        np.set_printoptions(3, suppress=True)

        io_it = self.get_io_iterator()

        for input_path in io_it:
            logging.info('------------------------------------------')
            logging.info('Looking at {0}'.format(input_path))
            logging.info('------------------------------------------')

            ipath_lower = input_path.lower()

            if ipath_lower.endswith('.nii') or ipath_lower.endswith('.nii.gz'):

                data, affine, img, vox_sz, affcodes = load_nifti(
                    input_path,
                    return_img=True,
                    return_voxsize=True,
                    return_coords=True)
                logging.info('Data size {0}'.format(data.shape))
                logging.info('Data type {0}'.format(data.dtype))
                logging.info('Data min {0} max {1} avg {2}'
                             .format(data.min(), data.max(), data.mean()))
                logging.info('2nd percentile {0} 98th percentile {1}'
                             .format(np.percentile(data, 2),
                                     np.percentile(data, 98)))
                logging.info('Native coordinate system {0}'
                             .format(''.join(affcodes)))
                logging.info('Affine to RAS1mm \n{0}'.format(affine))
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

    def load_module(self, module_path):
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

        Parameters
        ----------
        data_names : variable string
            Any number of Nifti1, bvals or bvecs files.
        out_dir : string, optional
            Output directory. Default: dipy home folder (~/.dipy)

        """
        if out_dir:
            dipy_home = os.environ.get('DIPY_HOME', None)
            os.environ['DIPY_HOME'] = out_dir

        fetcher_module = self.load_module('dipy.data.fetcher')

        available_data = dict([(name.replace('fetch_', ''), func)
                               for name, func in getmembers(fetcher_module,
                                                            isfunction)
                               if name.lower().startswith("fetch_")
                               if not len(getfullargspec(func).args)])

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
