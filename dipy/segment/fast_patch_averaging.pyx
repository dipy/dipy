cimport cython
from cython.view cimport array as cvarray
from libc.math cimport sqrt, exp
import numpy as np
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _modify_patch(double[:, :, :] inp, int c0, int c1, int c2,
                        int pr, double[:, :, :] out) nogil:
    """
    Normalize the patch with the sum of the patch
    """
    cdef:
        cnp.npy_intp i, j, k
        double patch_sum = 0.0

    for i in range(c0 - pr, c0 + pr + 1):
        for j in range(c1 - pr, c1 + pr + 1):
            for k in range(c2 - pr, c2 + pr + 1):
                patch_sum += inp[i, j, k]

    for i in range(c0 - pr, c0 + pr + 1):
        for j in range(c1 - pr, c1 + pr + 1):
            for k in range(c2 - pr, c2 + pr + 1):
                if patch_sum != 0:
                    out[pr - c0 + i, pr - c1 + j,
                        pr - c2 + k] = inp[i, j, k] / patch_sum
                else:
                    out[pr - c0 + i, pr - c1 + j,
                        pr - c2 + k] = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _compute_weight(double[:, :, :] p1, double[:, :, :] p2,
                          double h) nogil:
    """
    Computing weight between two 3D patches
    """
    cdef cnp.npy_intp i, j, k
    cdef double temp = 0.0
    cdef double out = 0.0

    for i in range(p1.shape[0]):
        for j in range(p1.shape[1]):
            for k in range(p1.shape[2]):
                temp += (p1[i, j, k] - p2[i, j, k]) * (p1[i, j, k] - p2[i, j, k])

    out = exp( -1 * temp / (h * h))
    return out


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
        cnp.npy_intp total_radius = patch_radius + block_radius
        cnp.npy_intp i, j, k, i0, j0, k0
        double wt = 0.0
        double wtsum, wtavg
        double[:, :, :] output_data = np.zeros((n0, n1, n2), dtype=np.float64)
        double[:, :, :] output_mask = np.zeros((n0, n1, n2), dtype=np.float64)
        double[:, :, :] output_mask1 = np.zeros((n0, n1, n2), dtype=np.float64)
        double[:, :, :] cen_patch = np.zeros((patch_size, patch_size, patch_size), dtype=np.float64)
        double[:, :, :] nl_patch = np.zeros((patch_size, patch_size, patch_size), dtype=np.float64)

    with nogil:
        for i in range(total_radius, n0 - total_radius):
            with gil:
                print(i)
            for j in range(total_radius, n1 - total_radius):
                for k in range(total_radius, n2 - total_radius):
                    wtsum = 0.0
                    wtavg = 0.0
                    _modify_patch(input_data, i, j, k, patch_radius, cen_patch)

                    for i0 in range(i - block_radius, i + block_radius + 1):
                        for j0 in range(
                                j - block_radius, j + block_radius + 1):
                            for k0 in range(
                                    k - block_radius, k + block_radius + 1):

                                _modify_patch(
                                    transformed_data, i0, j0, k0, patch_radius, nl_patch)
                                
                                wt = _compute_weight(cen_patch, nl_patch, parameter)

                                #with gil:
                                #    wt = np.exp(-np.sum((np.array(cen_patch) - \
                                #            np.array(nl_patch))**2) / parameter**2)
                                    
                                wtsum += wt
                                wtavg += wt * transformed_mask[i0, j0, k0]

                    if wtsum == 0:
                        output_mask[i, j, k] = 0
                    else:
                        output_mask[i, j, k] = wtavg / wtsum

        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                    if(output_mask[i, j, k] > threshold):
                        output_mask1[i, j, k] = 1
                        output_data[i, j, k] = input_data[i, j, k]
                    else: 
                        output_mask[i, j, k] = 0

    return [np.array(output_data), np.array(output_mask)]
