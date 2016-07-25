cimport cython
from cython.view cimport array as cvarray
from libc.math cimport sqrt, exp
import numpy as np
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _modify_patch(double[:, :, :] input_data, int c0, int c1, int c2,
                        int patch_radius, double[:, :, :] out) nogil:
    """
    Normalize the patch with the sum of the patch
    """
    cdef cnp.npy_intp i, j, k
    cdef double patch_sum = 0

    for i in range(c0 - patch_radius, c0 + patch_radius + 1):
        for j in range(c1 - patch_radius, c1 + patch_radius + 1):
            for k in range(c2 - patch_radius, c2 + patch_radius + 1):
                patch_sum += input_data[i, j, k]

    for i in range(c0 - patch_radius, c0 + patch_radius + 1):
        for j in range(c1 - patch_radius, c1 + patch_radius + 1):
            for k in range(c2 - patch_radius, c2 + patch_radius + 1):
                out[patch_radius - c0 + i, patch_radius - c1 + i,
                    patch_radius - c2 + i] = input_data[i, j, k] / patch_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fast_patch_averaging(double[:, :, :] input_data, double[:, :, :] transformed_data, double[:, :, :] transformed_mask,
                         int patch_radius, int block_radius, double parameter,
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

    cdef:
        cnp.npy_intp n0 = input_data.shape[0]
        cnp.npy_intp n1 = input_data.shape[1]
        cnp.npy_intp n2 = input_data.shape[2]
        cnp.npy_intp patch_size = 2 * patch_radius + 1
        cnp.npy_intp block_size = 2 * block_radius + 1
        cnp.npy_intp total_radius = patch_radius + block_radius
        cnp.npy_intp i, j, k, i0, j0, k0
        double[:, :, :] output_data = np.zeros((n0, n1, n2), dtype=np.float64)
        double[:, :, :] output_mask = np.zeros((n0, n1, n2), dtype=np.float64)
        double[:, :, :] cen_patch = np.zeros((patch_radius, patch_radius, patch_radius), dtype=np.float64)
        double[:, :, :] nl_patch = np.zeros((patch_radius, patch_radius, patch_radius), dtype=np.float64)

    with nogil:
        for i in range(total_radius, n0 - total_radius, 2):
            for j in range(total_radius, n1 - total_radius, 2):
                for k in range(total_radius, n2 - total_radius, 2):

                    _modify_patch(input_data, i, j, k, patch_radius, cen_patch)

                    for i0 in range(i - block_radius, i + block_radius + 1):
                        for j0 in range(
                                j - block_radius, j + block_radius + 1):
                            for k0 in range(
                                    k - block_radius, k + block_radius + 1):

                                _modify_patch(
                                    transformed_data, i0, j0, k0, patch_radius, nl_patch)

    return input_data
