# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: wraparound=False
"""
================================
Non-Local Means Denoising
================================

"""
import numpy as np
cimport numpy as cnp

cimport cython

from cython.parallel import parallel, prange, threadid
from dipy.utils.omp import determine_num_threads
from dipy.utils.omp cimport set_num_threads, restore_default_num_threads

from libc.math cimport sqrt, exp
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy



def nlmeans_3d_classic(arr, mask=None, sigma=None, patch_radius=1,
               block_radius=5, rician=True, num_threads=None):
    """ Non-local means for denoising 3D images

    Parameters
    ----------
    arr : 3D ndarray
        The array to be denoised
    mask : 3D ndarray
    sigma : float or 3D array
        standard deviation of the noise estimated from the data
    patch_radius : int
        patch size is ``2 x patch_radius + 1``. Default is 1.
    block_radius : int
        block size is ``2 x block_radius + 1``. Default is 5.
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.
    num_threads : int, optional
        Number of threads to be used for OpenMP parallelization. If None
        (default) the value of OMP_NUM_THREADS environment variable is used
        if it is set, otherwise all available threads are used. If < 0 the
        maximal number of threads minus $|num_threads + 1|$ is used (enter -1 to
        use as many threads as possible). 0 raises an error.

    Returns
    -------
    denoised_arr : ndarray
        the denoised ``arr`` which has the same shape as ``arr``.
    """

    if arr.ndim != 3:
        raise ValueError('data needs to be a 3D ndarray', arr.shape)

    if mask is None:
        mask = np.ones(arr.shape, dtype='f8')
    else:
        mask = np.ascontiguousarray(mask, dtype='f8')

    if mask.ndim != 3:
        raise ValueError('mask needs to be a 3D ndarray', mask.shape)

    if sigma.ndim != 3:
        raise ValueError('sigma needs to be a 3D ndarray', sigma.shape)

    arr = np.ascontiguousarray(arr, dtype='f8')
    arr = add_padding_reflection(arr, block_radius)
    mask = add_padding_reflection(mask.astype('f8'), block_radius)
    sigma = np.ascontiguousarray(sigma, dtype='f8')
    sigma = add_padding_reflection(sigma.astype('f8'), block_radius)
    arrnlm = _nlmeans_3d(arr, mask, sigma, patch_radius, block_radius,
                         rician, num_threads)

    return remove_padding(arrnlm, block_radius)


def _nlmeans_3d(double[:, :, ::1] arr, double[:, :, ::1] mask,
                double[:, :, ::1] sigma, patch_radius=1, block_radius=5,
                rician=True, num_threads=None):
    """ This algorithm denoises the value of every voxel (i, j, k) by
    calculating a weight between a moving 3D patch and a static 3D patch
    centered at (i, j, k). The moving patch can only move inside a
    3D block.
    """

    cdef:
        cnp.npy_intp i, j, k, I, J, K
        double[:, :, ::1] out = np.zeros_like(arr)
        double summ = 0
        cnp.npy_intp P = patch_radius
        cnp.npy_intp B = block_radius
        int threads_to_use = -1

    I = arr.shape[0]
    J = arr.shape[1]
    K = arr.shape[2]

    threads_to_use = determine_num_threads(num_threads)
    set_num_threads(threads_to_use)

    # move the block
    with nogil, parallel():

        for i in prange(B, I - B, schedule="dynamic"):
            for j in range(B, J - B):
                for k in range(B, K - B):

                    if mask[i, j, k] == 0:
                        continue

                    out[i, j, k] = process_block(arr, i, j, k, B, P, sigma)

    if num_threads is not None:
        restore_default_num_threads()

    new = np.asarray(out)

    if rician:
        new -= 2 * np.asarray(sigma)**2
        new[new < 0] = 0

    return np.sqrt(new)


cdef double process_block(double[:, :, ::1] arr,
                          cnp.npy_intp i, cnp.npy_intp j, cnp.npy_intp k,
                          cnp.npy_intp B, cnp.npy_intp P, double[:, :, ::1] sigma) nogil:
    """ Process the block with center at (i, j, k)

    Parameters
    ----------
    arr : 3D array
        C contiguous array of doubles
    i, j, k : int
        center of block
    B : int
        block radius
    P : int
        patch radius
    sigma : 3D array
        local noise standard deviation

    Returns
    -------
    new_value : double
    """

    cdef:
        cnp.npy_intp m, n, o, M, N, O, patch_vol_size, a, b, c, cnt, step
        double summ, d, w, sumw, sum_out, x, sigm
        double * W
        double * cache
        double * sigma_block
        double denom
        cnp.npy_intp BS = B * 2 + 1

    cnt = 0
    sumw = 0
    patch_vol_size = (P + P + 1) * (P + P + 1) * (P + P + 1)

    W = <double *> malloc(BS * BS * BS * sizeof(double))
    cache = <double *> malloc(BS * BS * BS * sizeof(double))
    sigma_block = <double *> malloc(BS * BS * BS * sizeof(double))

    # (i, j, k) coordinates are the center of the static patch
    # copy block in cache
    copy_block_3d(cache, BS, BS, BS, arr, i - B, j - B, k - B)
    copy_block_3d(sigma_block, BS, BS, BS, sigma, i - B, j - B, k - B)

    # calculate weights between the central patch and the moving patch in block
    # (m, n, o) coordinates are the center of the moving patch
    # (a, b, c) run inside both patches
    for m in range(P, BS - P):
        for n in range(P, BS - P):
            for o in range(P, BS - P):

                summ = 0
                sigm = 0

                # calculate square distance
                for a in range(-P, P + 1):
                    for b in range(-P, P + 1):
                        for c in range(-P, P + 1):

                            # this line takes most of the time! mem access
                            d = cache[(B + a) * BS * BS + (B + b) * BS + (B + c)] - \
                                cache[(m + a) * BS * BS + (n + b) * BS + (o + c)]
                            summ += d * d
                            sigm += sigma_block[(m + a) *
                                                BS * BS + (n + b) * BS + (o + c)]

                denom = sqrt(2) * (sigm / patch_vol_size)**2
                w = exp(-(summ / patch_vol_size) / denom)
                sumw += w
                W[cnt] = w
                cnt += 1

    cnt = 0
    sum_out = 0

    # calculate normalized weights and sums of the weights with the positions
    # of the patches
    for m in range(P, BS - P):
        for n in range(P, BS - P):
            for o in range(P, BS - P):

                if sumw > 0:
                    w = W[cnt] / sumw
                else:
                    w = 0

                x = cache[m * BS * BS + n * BS + o]
                sum_out += w * x * x
                cnt += 1

    free(W)
    free(cache)
    free(sigma_block)

    return sum_out


def add_padding_reflection(double[:, :, ::1] arr, padding):
    cdef:
        double[:, :, ::1] final
        cnp.npy_intp i, j, k
        cnp.npy_intp B = padding
        cnp.npy_intp[::1] indices_i = correspond_indices(arr.shape[0], padding)
        cnp.npy_intp[::1] indices_j = correspond_indices(arr.shape[1], padding)
        cnp.npy_intp[::1] indices_k = correspond_indices(arr.shape[2], padding)

    final = np.zeros(
        np.array(
            (arr.shape[0],
             arr.shape[1],
             arr.shape[2])) +
        2 *
        padding)

    for i in range(final.shape[0]):
        for j in range(final.shape[1]):
            for k in range(final.shape[2]):
                final[i, j, k] = arr[indices_i[i], indices_j[j], indices_k[k]]

    return final


def correspond_indices(dim_size, padding):
    return np.ascontiguousarray(np.hstack((np.arange(1,
                                                     padding + 1)[::-1],
                                           np.arange(dim_size),
                                           np.arange(dim_size - padding - 1,
                                                     dim_size - 1)[::-1])),
                                dtype=np.intp)


def remove_padding(arr, padding):
    shape = arr.shape
    return arr[padding:shape[0] - padding,
               padding:shape[1] - padding,
               padding:shape[2] - padding]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef cnp.npy_intp copy_block_3d(double * dest,
                                cnp.npy_intp I,
                                cnp.npy_intp J,
                                cnp.npy_intp K,
                                double[:, :, ::1] source,
                                cnp.npy_intp min_i,
                                cnp.npy_intp min_j,
                                cnp.npy_intp min_k) nogil:

    cdef cnp.npy_intp i, j

    for i in range(I):
        for j in range(J):
            memcpy(&dest[i * J * K  + j * K], &source[i + min_i, j + min_j, min_k], K * sizeof(double))

    return 1


# ==== BLOCKWISE NON-LOCAL MEANS IMPLEMENTATION ====

cdef inline int _int_max(int a, int b):
    return a if a >= b else b
cdef inline int _int_min(int a, int b):
    return a if a <= b else b


def _firdn_vector(double[:] f, double[:] h, double[:] out):
    cdef int n = len(f)
    cdef int klen = len(h)
    cdef int outLen = (n + klen) // 2
    cdef double ss
    cdef int i, k, limInf, limSup, x = 0, ox = 0, ks = 0
    for i in range(outLen):
        ss = 0
        limInf = _int_max(0, x - klen + 1)
        limSup = 1 + _int_min(n - 1, x)
        ks = limInf
        for k in range(limInf, limSup):
            ss += f[ks] * h[x - k]
            ks += 1
        out[ox] = ss
        x += 2
        ox += 1


def _upfir_vector(double[:] f, double[:] h, double[:] out):
    cdef int n = f.shape[0]
    cdef int klen = h.shape[0]
    cdef int outLen = 2 * n + klen - 2
    cdef int x, limInf, limSup, k, ks
    cdef double ss
    for x in range(outLen):
        limInf = _int_max(0, x - klen + 1)
        if limInf % 2 == 1:
            limInf += 1
        limSup = _int_min(2 * (n - 1), x)
        if limSup % 2 == 1:
            limSup -= 1
        ss = 0
        k = limInf
        ks = limInf // 2
        while k <= limSup:
            ss += f[ks] * h[x - k]
            k += 2
            ks += 1
        out[x] = ss


def _firdn_matrix(double[:, :] F, double[:] h, double[:, :] out):
    cdef int n = F.shape[0]
    cdef int m = F.shape[1]
    cdef int j
    for j in range(m):
        _firdn_vector(F[:, j], h, out[:, j])


def _upfir_matrix(double[:, :] F, double[:] h, double[:, :] out):
    cdef int n = F.shape[0]
    cdef int m = F.shape[1]
    for j in range(m):
        _upfir_vector(F[:, j], h, out[:, j])


cpdef firdn(double[:, :] image, double[:] h):
    """
    Applies the filter given by the convolution kernel 'h' columnwise to
    'image', then subsamples by 2. This is a special case of the matlab's
    'upfirdn' function, ported to python. Returns the filtered image.

    Parameters
    ----------
    image: 2D array of doubles
        the input image to be filtered
    h: double array
        the convolution kernel
    """
    nrows = image.shape[0]
    ncols = image.shape[1]
    ll = h.shape[0]
    cdef double[:, :] filtered = np.zeros(shape=((nrows + ll) // 2, ncols))
    _firdn_matrix(image, h, filtered)
    return filtered

cpdef upfir(double[:, :] image, double[:] h):
    """
    Upsamples the columns of the input image by 2, then applies the
    convolution kernel 'h' (again, columnwise). This is a special case of the
    matlab's 'upfirdn' function, ported to python. Returns the filtered image.

    Parameters
    ----------
    image: 2D array of doubles
        the input image to be filtered
    h: double array
        the convolution kernel
    """
    nrows = image.shape[0]
    ncols = image.shape[1]
    ll = h.shape[0]
    cdef double[:, :] filtered = np.zeros(shape=(2 * nrows + ll - 2, ncols))
    _upfir_matrix(image, h, filtered)
    return filtered


cdef void _average_block(double[:, :, :] image, int center_y, int center_x, int center_z,
                         double[:, :, :] weighted_average, double weight) noexcept nogil:
    """
    Compute the weighted average of the patches in a blockwise manner.

    This function accumulates weighted squared intensities from a block centered
    at (center_y, center_x, center_z) into the weighted_average array.

    Parameters
    ----------
    image : 3D array of doubles, shape (height, width, depth)
        Input image with standard array indexing [y, x, z]
    center_y : int
        Y coordinate (row index) of the center voxel
    center_x : int
        X coordinate (column index) of the center voxel
    center_z : int
        Z coordinate (depth index) of the center voxel
    weighted_average : 3D array of doubles
        Accumulator for weighted averages
    weight : double
        Weight for the weighted averaging
    """
    cdef int offset_y, offset_x, offset_z
    cdef int voxel_y, voxel_x, voxel_z
    cdef int is_outside_bounds
    cdef int block_radius = weighted_average.shape[0] // 2
    cdef int img_height = image.shape[0]
    cdef int img_width = image.shape[1]
    cdef int img_depth = image.shape[2]

    for offset_y in range(weighted_average.shape[0]):
        for offset_x in range(weighted_average.shape[1]):
            for offset_z in range(weighted_average.shape[2]):
                voxel_y = center_y + offset_y - block_radius
                voxel_x = center_x + offset_x - block_radius
                voxel_z = center_z + offset_z - block_radius

                is_outside_bounds = 0
                if voxel_y < 0 or voxel_y >= img_height:
                    is_outside_bounds = 1
                if voxel_x < 0 or voxel_x >= img_width:
                    is_outside_bounds = 1
                if voxel_z < 0 or voxel_z >= img_depth:
                    is_outside_bounds = 1

                if is_outside_bounds == 1:
                    # Use center voxel value for out-of-bounds locations
                    weighted_average[offset_y, offset_x, offset_z] += weight * (image[center_y, center_x, center_z]**2)
                else:
                    weighted_average[offset_y, offset_x, offset_z] += weight * (image[voxel_y, voxel_x, voxel_z]**2)


cdef void _value_block(double[:, :, :] denoised_estimate, double[:, :, :] weight_count,
                       int center_y, int center_x, int center_z,
                       double[:, :, :] weighted_average, double total_weight,
                       double noise_variance_doubled, int is_rician) noexcept nogil:
    """
    Computes the final denoised estimate for a block and accumulates it.

    This function processes each voxel in a block centered at (center_y, center_x, center_z),
    computes the denoised value from weighted averages, and accumulates the results.

    Parameters
    ----------
    denoised_estimate : 3D array of doubles, shape (height, width, depth)
        Accumulator for denoised intensity estimates
    weight_count : 3D array of doubles, shape (height, width, depth)
        Count of how many times each voxel has been processed
    center_y : int
        Y coordinate (row index) of the block center
    center_x : int
        X coordinate (column index) of the block center
    center_z : int
        Z coordinate (depth index) of the block center
    weighted_average : 3D array of doubles
        Weighted average intensities for the block
    total_weight : double
        Sum of all weights used in the averaging
    noise_variance_doubled : double
        2 * sigma^2 for Rician noise correction
    is_rician : int
        1 if Rician noise model, 0 for Gaussian
    """
    cdef int offset_y, offset_x, offset_z
    cdef int voxel_y, voxel_x, voxel_z
    cdef int is_outside_bounds
    cdef int block_radius = weighted_average.shape[0] // 2
    cdef int img_height = denoised_estimate.shape[0]
    cdef int img_width = denoised_estimate.shape[1]
    cdef int img_depth = denoised_estimate.shape[2]
    cdef double current_estimate, denoised_intensity, current_count

    for offset_y in range(weighted_average.shape[0]):
        for offset_x in range(weighted_average.shape[1]):
            for offset_z in range(weighted_average.shape[2]):
                voxel_y = center_y + offset_y - block_radius
                voxel_x = center_x + offset_x - block_radius
                voxel_z = center_z + offset_z - block_radius

                is_outside_bounds = 0
                if voxel_y < 0 or voxel_y >= img_height:
                    is_outside_bounds = 1
                if voxel_x < 0 or voxel_x >= img_width:
                    is_outside_bounds = 1
                if voxel_z < 0 or voxel_z >= img_depth:
                    is_outside_bounds = 1

                if is_outside_bounds == 0:
                    current_estimate = denoised_estimate[voxel_y, voxel_x, voxel_z]

                    # Compute denoised intensity from weighted average
                    if total_weight > 0:
                        if is_rician:
                            denoised_intensity = (weighted_average[offset_y, offset_x, offset_z] / total_weight) - noise_variance_doubled
                        else:
                            denoised_intensity = weighted_average[offset_y, offset_x, offset_z] / total_weight

                        if denoised_intensity > 0:
                            denoised_intensity = sqrt(denoised_intensity)
                        else:
                            denoised_intensity = 0.0
                    else:
                        denoised_intensity = 0.0

                    # Accumulate results
                    current_count = weight_count[voxel_y, voxel_x, voxel_z]
                    denoised_estimate[voxel_y, voxel_x, voxel_z] = current_estimate + denoised_intensity
                    weight_count[voxel_y, voxel_x, voxel_z] = current_count + 1.0


cdef double _patch_distance(double[:, :, :] image,
                           int patch1_center_y, int patch1_center_x, int patch1_center_z,
                           int patch2_center_y, int patch2_center_x, int patch2_center_z,
                           int patch_radius) nogil:
    """
    Computes the squared distance between two cubic patches in the image.

    Uses reflection padding at boundaries to handle edge cases properly.
    The distance is normalized by the number of voxels compared.

    Parameters
    ----------
    image : 3D array of doubles, shape (height, width, depth)
        Input image with standard array indexing [y, x, z]
    patch1_center_y, patch1_center_x, patch1_center_z : int
        Y, X, Z coordinates of the first patch center
    patch2_center_y, patch2_center_x, patch2_center_z : int
        Y, X, Z coordinates of the second patch center
    patch_radius : int
        Radius of the cubic patches (patch size = 2*radius + 1)

    Returns
    -------
    double
        Mean squared distance between the two patches
    """
    cdef double squared_distance_sum = 0.0
    cdef int voxel_count = 0
    cdef int offset_y, offset_x, offset_z
    cdef int voxel1_y, voxel1_x, voxel1_z
    cdef int voxel2_y, voxel2_x, voxel2_z
    cdef int img_height = image.shape[0]
    cdef int img_width = image.shape[1]
    cdef int img_depth = image.shape[2]
    cdef double intensity_diff

    for offset_y in range(-patch_radius, patch_radius + 1):
        for offset_x in range(-patch_radius, patch_radius + 1):
            for offset_z in range(-patch_radius, patch_radius + 1):
                # Calculate voxel coordinates for both patches
                voxel1_y = patch1_center_y + offset_y
                voxel1_x = patch1_center_x + offset_x
                voxel1_z = patch1_center_z + offset_z

                voxel2_y = patch2_center_y + offset_y
                voxel2_x = patch2_center_x + offset_x
                voxel2_z = patch2_center_z + offset_z

                # Apply reflection padding for out-of-bounds coordinates
                # Y dimension (height)
                if voxel1_y < 0:
                    voxel1_y = -voxel1_y
                elif voxel1_y >= img_height:
                    voxel1_y = 2 * img_height - voxel1_y - 1

                if voxel2_y < 0:
                    voxel2_y = -voxel2_y
                elif voxel2_y >= img_height:
                    voxel2_y = 2 * img_height - voxel2_y - 1

                # X dimension (width)
                if voxel1_x < 0:
                    voxel1_x = -voxel1_x
                elif voxel1_x >= img_width:
                    voxel1_x = 2 * img_width - voxel1_x - 1

                if voxel2_x < 0:
                    voxel2_x = -voxel2_x
                elif voxel2_x >= img_width:
                    voxel2_x = 2 * img_width - voxel2_x - 1

                # Z dimension (depth)
                if voxel1_z < 0:
                    voxel1_z = -voxel1_z
                elif voxel1_z >= img_depth:
                    voxel1_z = 2 * img_depth - voxel1_z - 1

                if voxel2_z < 0:
                    voxel2_z = -voxel2_z
                elif voxel2_z >= img_depth:
                    voxel2_z = 2 * img_depth - voxel2_z - 1

                # Compute squared difference
                intensity_diff = image[voxel1_y, voxel1_x, voxel1_z] - image[voxel2_y, voxel2_x, voxel2_z]
                squared_distance_sum += intensity_diff * intensity_diff
                voxel_count += 1

    return squared_distance_sum / voxel_count


cdef void _process_block_complete(double[:, :, :] image,
                                double[:, :, :] mask,
                                double[:, :, :] local_means,
                                double[:, :, :] local_variances,
                                double[:, :, :] accumulated_estimates,
                                double[:, :, :] weight_counts,
                                double[:, :, :] workspace,
                                int center_y, int center_x, int center_z,
                                int patch_radius, int block_radius,
                                double filtering_strength,
                                double noise_variance_doubled, int is_rician,
                                double epsilon, double mean_ratio_threshold,
                                double variance_ratio_min,
                                int img_height, int img_width, int img_depth) noexcept nogil:
    """
    Process a complete block including weight computation and final application.

    This function avoids reduction variables by computing and applying weights
    in a single pass without intermediate storage.
    """
    cdef double current_mean = local_means[center_y, center_x, center_z]
    cdef double current_variance = local_variances[center_y, center_x, center_z]
    cdef double accumulator_weight = 0.0
    cdef double max_weight = 0.0
    cdef double similarity_weight, patch_distance, mean_ratio, variance_ratio
    cdef double neighbor_mean, neighbor_variance
    cdef int neighbor_y, neighbor_x, neighbor_z
    cdef int offset_y, offset_x, offset_z
    cdef double variance_ratio_max = 1.0 / variance_ratio_min

    # Handle low signal/noise regions with simple averaging
    if current_mean <= epsilon or current_variance <= epsilon:
        max_weight = 1.0
        _average_block(image, center_y, center_x, center_z, workspace, max_weight)
        accumulator_weight = max_weight
    else:
        # Search for similar patches in neighborhood
        for offset_z in range(-patch_radius, patch_radius + 1):
            neighbor_z = center_z + offset_z
            if neighbor_z < 0 or neighbor_z >= img_depth:
                continue

            for offset_x in range(-patch_radius, patch_radius + 1):
                neighbor_x = center_x + offset_x
                if neighbor_x < 0 or neighbor_x >= img_width:
                    continue

                for offset_y in range(-patch_radius, patch_radius + 1):
                    neighbor_y = center_y + offset_y
                    if neighbor_y < 0 or neighbor_y >= img_height:
                        continue
                    if neighbor_y == center_y and neighbor_x == center_x and neighbor_z == center_z:
                        continue
                    if mask[neighbor_y, neighbor_x, neighbor_z] == 0:
                        continue

                    neighbor_mean = local_means[neighbor_y, neighbor_x, neighbor_z]
                    neighbor_variance = local_variances[neighbor_y, neighbor_x, neighbor_z]

                    # Skip patches with insufficient signal
                    if neighbor_mean <= epsilon or neighbor_variance <= epsilon:
                        continue

                    # Pre-filtering based on statistical similarity
                    mean_ratio = current_mean / neighbor_mean
                    variance_ratio = current_variance / neighbor_variance

                    if (mean_ratio < mean_ratio_threshold or mean_ratio > (1.0 / mean_ratio_threshold) or
                        variance_ratio < variance_ratio_min or variance_ratio > variance_ratio_max):
                        continue

                    # Compute patch distance
                    patch_distance = _patch_distance(image,
                                                    center_y, center_x, center_z,
                                                    neighbor_y, neighbor_x, neighbor_z,
                                                    block_radius)

                    # Compute similarity weight using original nlmeans formula
                    # Original: w = exp(-d / (h * h)) where h is the noise sigma
                    similarity_weight = exp(-patch_distance / filtering_strength)

                    if similarity_weight > max_weight:
                        max_weight = similarity_weight

                    # Accumulate weighted averages directly
                    _average_block(image, neighbor_y, neighbor_x, neighbor_z, workspace, similarity_weight)
                    accumulator_weight += similarity_weight

    # Apply accumulated weighted averages to final estimate
    if accumulator_weight > 0.0:
        _value_block(accumulated_estimates, weight_counts,
                   center_y, center_x, center_z, workspace,
                   accumulator_weight, noise_variance_doubled, is_rician)


cdef inline void _clear_workspace(double[:, :, :] workspace) noexcept nogil:
    """
    Efficiently clears a 3D workspace array by setting all values to zero.

    Parameters
    ----------
    workspace : 3D array of doubles
        The workspace array to clear
    """
    cdef int i, j, k
    cdef int size_i = workspace.shape[0]
    cdef int size_j = workspace.shape[1]
    cdef int size_k = workspace.shape[2]

    for i in range(size_i):
        for j in range(size_j):
            for k in range(size_k):
                workspace[i, j, k] = 0.0


cdef double _local_mean(double[:, :, :] image, int center_y, int center_x, int center_z, int patch_radius) nogil:
    """
    Computes the local mean of a cubic patch centered at (center_y, center_x, center_z).

    Uses reflection padding for boundary handling.

    Parameters
    ----------
    image : 3D array of doubles, shape (height, width, depth)
        Input image with standard array indexing [y, x, z]
    center_y, center_x, center_z : int
        Y, X, Z coordinates of the patch center
    patch_radius : int
        Radius of the patch (patch size = 2*radius + 1 in each dimension)

    Returns
    -------
    double
        Mean intensity value of the patch
    """
    cdef int img_height = image.shape[0]
    cdef int img_width = image.shape[1]
    cdef int img_depth = image.shape[2]
    cdef double intensity_sum = 0.0
    cdef int voxel_count = 0
    cdef int voxel_y, voxel_x, voxel_z
    cdef int offset_y, offset_x, offset_z

    for offset_y in range(-patch_radius, patch_radius + 1):
        for offset_x in range(-patch_radius, patch_radius + 1):
            for offset_z in range(-patch_radius, patch_radius + 1):
                voxel_y = center_y + offset_y
                voxel_x = center_x + offset_x
                voxel_z = center_z + offset_z

                # Apply reflection padding
                if voxel_y < 0:
                    voxel_y = -voxel_y
                elif voxel_y >= img_height:
                    voxel_y = 2 * img_height - voxel_y - 1

                if voxel_x < 0:
                    voxel_x = -voxel_x
                elif voxel_x >= img_width:
                    voxel_x = 2 * img_width - voxel_x - 1

                if voxel_z < 0:
                    voxel_z = -voxel_z
                elif voxel_z >= img_depth:
                    voxel_z = 2 * img_depth - voxel_z - 1

                intensity_sum += image[voxel_y, voxel_x, voxel_z]
                voxel_count += 1

    return intensity_sum / voxel_count


cdef double _local_variance(double[:, :, :] image, double mean_intensity,
                           int center_y, int center_x, int center_z, int patch_radius) nogil:
    """
    Computes the local variance of a cubic patch centered at (center_y, center_x, center_z).

    Uses reflection padding for boundary handling.

    Parameters
    ----------
    image : 3D array of doubles, shape (height, width, depth)
        Input image with standard array indexing [y, x, z]
    mean_intensity : double
        Pre-computed mean intensity of the patch
    center_y, center_x, center_z : int
        Y, X, Z coordinates of the patch center
    patch_radius : int
        Radius of the patch (patch size = 2*radius + 1 in each dimension)

    Returns
    -------
    double
        Variance of intensities in the patch
    """
    cdef int img_height = image.shape[0]
    cdef int img_width = image.shape[1]
    cdef int img_depth = image.shape[2]
    cdef double squared_diff_sum = 0.0
    cdef int voxel_count = 0
    cdef int voxel_y, voxel_x, voxel_z
    cdef int offset_y, offset_x, offset_z
    cdef double intensity_diff

    for offset_y in range(-patch_radius, patch_radius + 1):
        for offset_x in range(-patch_radius, patch_radius + 1):
            for offset_z in range(-patch_radius, patch_radius + 1):
                voxel_y = center_y + offset_y
                voxel_x = center_x + offset_x
                voxel_z = center_z + offset_z

                # Apply reflection padding
                if voxel_y < 0:
                    voxel_y = -voxel_y
                elif voxel_y >= img_height:
                    voxel_y = 2 * img_height - voxel_y - 1

                if voxel_x < 0:
                    voxel_x = -voxel_x
                elif voxel_x >= img_width:
                    voxel_x = 2 * img_width - voxel_x - 1

                if voxel_z < 0:
                    voxel_z = -voxel_z
                elif voxel_z >= img_depth:
                    voxel_z = 2 * img_depth - voxel_z - 1

                intensity_diff = image[voxel_y, voxel_x, voxel_z] - mean_intensity
                squared_diff_sum += intensity_diff * intensity_diff
                voxel_count += 1

    # Use N-1 for sample variance (Bessel's correction)
    if voxel_count > 1:
        return squared_diff_sum / (voxel_count - 1)
    else:
        return 0.0


def nlmeans_3d_blockwise(double[:, :, :] image, double[:, :, :] mask, int patch_radius, int block_radius, double noise_sigma, int is_rician, num_threads=None):
    """
    Non-Local Means Denoising Using Blockwise Averaging.

    This implementation uses efficient memory management and parallel processing
    for high-performance denoising of 3D images.

    See :footcite:p:`Coupe2008` and :footcite:p:`Coupe2012` for further details
    about the method.

    Parameters
    ----------
    image : 3D array of doubles, shape (height, width, depth)
        Input image corrupted with noise. Uses standard array indexing [y, x, z].
    mask : 3D array of doubles, shape (height, width, depth)
        Binary mask indicating which voxels to process (1) or skip (0).
    patch_radius : int
        Radius for local patch comparisons. Search is performed within a
        (2*patch_radius + 1)^3 neighborhood around each voxel.
    block_radius : int
        Radius for the blocks used in weighted averaging. Each block has
        size (2*block_radius + 1)^3.
    noise_sigma : double
        Estimated noise standard deviation in the input image.
    is_rician : int
        1 if Rician noise model should be used, 0 for Gaussian noise model.
    num_threads : int, optional
        Number of OpenMP threads to use. If None, uses all available threads.

    Returns
    -------
    denoised_image : 3D array of doubles
        Denoised output image with the same shape as input.

    References
    ----------
    .. footbibliography::
    """
    # Image dimensions
    cdef int img_height = image.shape[0]
    cdef int img_width = image.shape[1]
    cdef int img_depth = image.shape[2]

    # Handle num_threads parameter
    cdef int threads_to_use
    if num_threads is None:
        threads_to_use = determine_num_threads(None)
    else:
        threads_to_use = determine_num_threads(num_threads)

    # Algorithm parameters
    cdef double noise_variance_doubled = 2.0 * noise_sigma * noise_sigma
    cdef double filtering_strength = noise_sigma * noise_sigma
    cdef int block_size = 2 * block_radius + 1

    # Statistical filtering thresholds
    cdef double epsilon = 1e-5
    cdef double mean_ratio_threshold = 0.95
    cdef double variance_ratio_min = 0.5 + 1e-7
    cdef double variance_ratio_max = 1.0 / variance_ratio_min

    # Output arrays
    cdef double[:, :, :] denoised_image = np.zeros_like(image)
    cdef double[:, :, :] local_means = np.zeros_like(image)
    cdef double[:, :, :] local_variances = np.zeros_like(image)
    cdef double[:, :, :] accumulated_estimates = np.zeros_like(image)
    cdef double[:, :, :] weight_counts = np.zeros_like(image)

    # Thread-local working memory (allocated per thread to avoid race conditions)
    cdef double *thread_weighted_averages = NULL
    cdef int block_volume = block_size * block_size * block_size

    # Loop variables
    cdef int center_y, center_x, center_z
    cdef int neighbor_y, neighbor_x, neighbor_z
    cdef int offset_y, offset_x, offset_z
    cdef double mean_ratio, variance_ratio
    cdef double patch_distance, similarity_weight, max_weight, total_weight
    cdef double current_mean, current_variance
    cdef double neighbor_mean, neighbor_variance
    cdef int thread_id

    # Set number of threads
    set_num_threads(threads_to_use)

    # Pre-allocate weighted average workspace (one per potential thread)
    cdef int max_threads = threads_to_use
    cdef double[:, :, :, :] thread_workspaces = np.zeros((max_threads, block_size, block_size, block_size), dtype=np.float64)

    # Phase 1: Compute local statistics (means and variances) in parallel
    with nogil, parallel():
        for center_z in prange(img_depth, schedule="dynamic"):
            for center_x in range(img_width):
                for center_y in range(img_height):
                    if mask[center_y, center_x, center_z] > 0:
                        current_mean = _local_mean(image, center_y, center_x, center_z, 1)  # 3x3x3 patch
                        local_means[center_y, center_x, center_z] = current_mean
                        local_variances[center_y, center_x, center_z] = _local_variance(
                            image, current_mean, center_y, center_x, center_z, 1)

    # Phase 2: Process blocks with stride 2 for efficiency
    # Avoid reduction variables entirely by processing each block completely
    with nogil, parallel():
        for center_z in prange(0, img_depth, 2, schedule="dynamic"):
            thread_id = threadid()
            for center_x in range(0, img_width, 2):
                for center_y in range(0, img_height, 2):
                    if mask[center_y, center_x, center_z] == 0:
                        continue

                    current_mean = local_means[center_y, center_x, center_z]
                    current_variance = local_variances[center_y, center_x, center_z]

                    # Clear workspace for this thread
                    _clear_workspace(thread_workspaces[thread_id])

                    # Process this block completely in one go
                    _process_block_complete(image, mask, local_means, local_variances,
                                          accumulated_estimates, weight_counts,
                                          thread_workspaces[thread_id],
                                          center_y, center_x, center_z,
                                          patch_radius, block_radius, filtering_strength,
                                          noise_variance_doubled, is_rician,
                                          epsilon, mean_ratio_threshold, variance_ratio_min,
                                          img_height, img_width, img_depth)

    # Phase 3: Finalize denoised estimates
    with nogil, parallel():
        for center_z in prange(img_depth, schedule="static"):
            for center_x in range(img_width):
                for center_y in range(img_height):
                    if mask[center_y, center_x, center_z] == 0:
                        denoised_image[center_y, center_x, center_z] = 0.0
                    elif weight_counts[center_y, center_x, center_z] > 0:
                        denoised_image[center_y, center_x, center_z] = (
                            accumulated_estimates[center_y, center_x, center_z] /
                            weight_counts[center_y, center_x, center_z])
                    else:
                        # Fallback to original value if no similar patches found
                        denoised_image[center_y, center_x, center_z] = image[center_y, center_x, center_z]

    # Restore default thread count
    restore_default_num_threads()

    return np.asarray(denoised_image)


def nlmeans_3d(arr, mask=None, sigma=None, patch_radius=1,
               block_radius=5, rician=True, num_threads=None, method='classic'):
    """ Non-local means for denoising 3D images with selectable algorithm

    Parameters
    ----------
    arr : 3D ndarray
        The array to be denoised
    mask : 3D ndarray
    sigma : float or 3D array
        standard deviation of the noise estimated from the data
    patch_radius : int
        patch size is ``2 x patch_radius + 1``. Default is 1.
    block_radius : int
        block size is ``2 x block_radius + 1``. Default is 5.
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.
    num_threads : int, optional
        Number of threads to be used for OpenMP parallelization. If None
        (default) the value of OMP_NUM_THREADS environment variable is used
        if it is set, otherwise all available threads are used. If < 0 the
        maximal number of threads minus $|num_threads + 1|$ is used (enter -1 to
        use as many threads as possible). 0 raises an error.
    method : str, optional
        Algorithm method to use. 'classic' for original algorithm,
        'blockwise' for improved blockwise algorithm. Default is 'classic'.

    Returns
    -------
    denoised_arr : ndarray
        the denoised ``arr`` which has the same shape as ``arr``.
    """

    if arr.ndim != 3:
        raise ValueError('data needs to be a 3D ndarray', arr.shape)

    if mask is None:
        mask = np.ones(arr.shape, dtype='f8')
    else:
        mask = np.ascontiguousarray(mask, dtype='f8')

    if mask.ndim != 3:
        raise ValueError('mask needs to be a 3D ndarray', mask.shape)

    # Handle sigma validation based on method
    if method == 'classic':
        # Classic method requires 3D sigma array
        if not hasattr(sigma, 'ndim') or sigma.ndim != 3:
            raise ValueError('sigma needs to be a 3D ndarray for classic method', getattr(sigma, 'shape', type(sigma)))
    elif method == 'blockwise':
        # Blockwise method can accept scalar or array sigma
        if hasattr(sigma, 'ndim'):
            if sigma.ndim > 3:
                raise ValueError('sigma should be at most 3D for blockwise method', sigma.shape)
        # Scalar sigma is fine for blockwise method
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'classic' or 'blockwise'.")

    arr = np.ascontiguousarray(arr, dtype='f8')

    if method == 'classic':
        # Use original classic algorithm with padding
        arr = add_padding_reflection(arr, block_radius)
        mask = add_padding_reflection(mask.astype('f8'), block_radius)
        sigma = np.ascontiguousarray(sigma, dtype='f8')
        sigma = add_padding_reflection(sigma.astype('f8'), block_radius)
        arrnlm = _nlmeans_3d(arr, mask, sigma, patch_radius, block_radius,
                             rician, num_threads)
        return remove_padding(arrnlm, block_radius)
    elif method == 'blockwise':
        # Use new blockwise algorithm without padding
        # For blockwise, we need a scalar sigma but can accept arrays
        if hasattr(sigma, 'shape'):
            if sigma.shape == arr.shape:
                # 3D sigma array - take the mean for uniform noise estimation
                sigma_scalar = np.mean(sigma)
            elif sigma.ndim == 1:
                # 1D sigma array - take the mean (for 4D case handled above)
                sigma_scalar = np.mean(sigma)
            elif sigma.shape == ():
                # 0-D array (scalar in array form)
                sigma_scalar = float(sigma)
            else:
                raise ValueError(f'Invalid sigma shape {sigma.shape} for blockwise method')
        else:
            # Scalar sigma
            sigma_scalar = float(sigma)

        return nlmeans_3d_blockwise(arr, mask, patch_radius, block_radius,
                                    sigma_scalar, int(rician), num_threads)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'classic' or 'blockwise'.")
