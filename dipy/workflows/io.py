from __future__ import division, print_function, absolute_import

import os
import numpy as np
import logging
from dipy.io.image import load_nifti

from dipy.workflows.workflow import Workflow


class SummarizeData(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'summarize'
        
    
    def run(self, input_files, out_dir=''):

        """ Workflow for creating a binary mask

        Parameters
        ----------
        input_files : variable string
           Nifti1, bvals and bvecs files.
        out_dir : string, optional
           Output directory (default input file directory)
        """

        #io_it = self.get_io_iterator()
        #logging.basicConfig(format='',)

        for input_path in input_files:
            logging.info('Summarizing {0}'.format(input_path))
            if input_path.endswith('.nii') or input_path.endswith('.nii.gz'):
                
                data, affine, img, vox_sz, affcodes = load_nifti(
                    input_path,
                    return_img=True,
                    return_voxsize=True,
                    return_coords=True)
                logging.info('Data size {0}'.format(data.shape))
                logging.info('Native coordinate system {0}'.format(affcodes))
                np.set_printoptions(3, suppress=True)
                logging.info('Affine to RAS1mm \n{0}'.format(affine))
                logging.info('Voxel size {0}'.format(np.array(vox_sz)))
                np.sum(np.abs(np.diff(vox_sz)))
                np.set_printoptions()
                
            if os.path.basename(input_path).find('bvals') > -1:
                bvals = np.loadtxt(input_path)
                logging.info('Bvalues \n {0}'.format(bvals))
                bvals = np.array([0, 1000, 2000, 3000, 3040, 2060], dtype='f8')
                bvals2 = np.sort(bvals)
                diffbvals2 = np.sum(np.diff(bvals2) > 100)
                
                logging.info('Bvalues sorted \n {0}'.format(bvals2))
                logging.info('Number of shells {0}'.format(diffbvals2))
                
                
    