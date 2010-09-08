''' Load some modules
'''

import dipy.core.track_metrics as track_metrics
import dipy.core.track_propagation as track_propagation
import dipy.core.track_performance as track_performance
import dipy.io.track_volumes as track_volumes

from dipy.core.generalized_q_sampling import GeneralizedQSampling
from dipy.core.dti import Tensor
from dipy.core.track_propagation import FACT_Delta

'''
try:
    from nipy.neurospin.registration import register as volume_register
    from nipy.neurospin.registration import transform as volume_transform
    from nipy.neurospin.registration._registration import rotation_vec2mat as _rotation_vec2mat
    from nipy.neurospin.registration.affine import Affine as _affine
    
except ImportError:
    print('nipy registration is not available')
'''
    
try:
    from nibabel import load, save
except ImportError:
    raise ImportError('nibabel is not installed')

try:
    from nibabel.dicom.dicomreaders import read_mosaic_dir as load_dcm_dir
except ImportError:
    raise ImportError('nibabel.dicom.dicomreaders cannot be found')
