import numpy as np
import nibabel as nib

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
import dipy.align.vector_fields as vfu


def syn_registration(moving, static, metric=CCMetric(3), 
                     level_iters = [10, 10, 5], prealign=None):
    """
    Register a source image (moving) to a target image (static)
    
    Parameters
    ----------
    moving : nibabel.Nifti1Image class instance
        The source image to be registered
    static : nibabel.Nifti1Image class instance
        The target image for registration
    metric : SimilarityMetric object, optional
        the metric to be optimized. Default: CCMetric(3).
    level_iters : list of int
        the number of iterations at each level of the Gaussian Pyramid (the
        length of the list defines the number of pyramid levels to be
        used).

    Returns
    -------
    new_image : nibabel.Nifti1Image class instance has the data that was originally in  
    `moving`, warped towards the `static` space  (affine of the result is the same as the 
    target). 
    
    transform : (x, y, z, 3) array containing the warp field from the `moving` image to 
    the static image. Each entry is the 3D vector of the warp applied in that location.
    
    """
    static_data = static.get_data()
    moving_data = moving.get_data()
    static_affine = static.get_affine()
    moving_affine = moving.get_affine()
    
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    mapping = sdr.optimize(static_data, moving_data, static_grid2world=None, 
                            moving_grid2world=None, prealign=None)
    
    warped_moving = mapping.transform(moving_data)
    new_image = nib.Nifti1Image(warped_moving, static_affine)

    return new_image, np.array(mapping.forward)