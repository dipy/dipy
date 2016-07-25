cimport cython
from cython.view cimport array as cvarray
from libc.math cimport sqrt, exp
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fast_patch_averaging(double[:,:,:] input_data, double[:,:,:] transformed_data, double[:,:,:] transformed_mask,
                    	int patch_radius, int block_radius,double parameter,
                    	double threshold):

    """
    input_data : 3D ndarray
        The input data from which the brain has to be extracted
    transformed_data : 3D ndarray
        The template data which is registered to the input
        (of the same size as that of the input data)
    transformed_mask : 3D ndarray
        The binary mask of the transformed template
    patch_radius : integer
        The patch size which has to be taken around the voxels for weight computation
    block_radius : integer
        Defining the neighbourhood around the voxel for patch wise similarity searching
    parameter : Double
        Adaptive parameter governing the weights for similar patches
    threshold : Double
        The threshold between 0 to 1 which decides the erosion of the mask boundary
    """

    return input_data
