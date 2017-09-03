from __future__ import division, print_function, absolute_import

import os
import numpy as np
import logging
logging.basicConfig(format='%(message)s',
                    level=logging.DEBUG)
#logging.basicConfig(format='%(levelname)s:%(message)s',
#                    level=logging.DEBUG)
from dipy.io.image import load_nifti

from dipy.workflows.workflow import Workflow


class IoInfo(Workflow):
    
    @classmethod
    def get_short_name(cls):
        return 'io_ino'
            
    def run(self, input_files, b0_threshold=50, bshell_thr=100):

        """ Workflow for creating a binary mask

        Parameters
        ----------
        input_files : variable string
            Nifti1, bvals and bvecs files.
        b0_threshold : float, optional
            (default 50)
        bshell_thr : float, optional
            Threshold for distinguishing b-values in different shells 
            (default 100)
        """
        
        np.set_printoptions(3, suppress=True)
        
        for input_path in input_files:
            logging.info('\nSummarizing {0}'.format(input_path))
            
            if input_path.endswith('.nii') or input_path.endswith('.nii.gz'):
                
                data, affine, img, vox_sz, affcodes = load_nifti(
                    input_path,
                    return_img=True,
                    return_voxsize=True,
                    return_coords=True)
                logging.info('Data size {0}'.format(data.shape))
                logging.info('Native coordinate system {0}'
                             .format(''.join(affcodes)))
                logging.info('Affine to RAS1mm \n{0}'.format(affine))
                logging.info('Voxel size {0}'.format(np.array(vox_sz)))
                if np.sum(np.abs(np.diff(vox_sz))) > 0.1:
                    msg = \
                    'Voxel size is not isotropic. Recommending reslicing.'
                    logging.warning(msg)
                                
            if os.path.basename(input_path).find('bval') > -1:
                bvals = np.loadtxt(input_path)
                logging.info('Bvalues \n{0}'.format(bvals))
                logging.info('Total number of bvalues {}'.format(len(bvals)))
                shells = np.sum(np.diff(np.sort(bvals)) > bshell_thr)
                logging.info('Number of shells {0}'.format(shells))
                logging.info('Number of b0s {0} using b0_threshold {1}'
                             .format(np.sum(bvals <= b0_threshold),
                                     b0_threshold))
                
            if os.path.basename(input_path).find('bvec') > -1:
                
                bvecs = np.loadtxt(input_path)
                logging.info('Bvectors shape on disk is {0}'
                             .format(bvecs.shape))
                rows, cols = bvecs.shape
                if rows < cols:
                    logging.info('Bvectors are \n{0}'.format(bvecs.T))
                else:
                    logging.info('Bvectors are \n{0}'.format(bvecs))
        
        np.set_printoptions()
                
                