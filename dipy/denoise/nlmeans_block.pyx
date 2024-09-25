cimport cython
from cython.view cimport array as cvarray
from libc.math cimport sqrt, exp
import numpy as np
cimport numpy as cnp

__all__ = ['firdn', 'upfir', 'nlmeans_block']

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _average_block(double[:, :, :] ima, int x, int y, int z,
                         double[:, :, :] average, double weight) noexcept nogil:
    """
    Computes the weighted average of the patches in a blockwise manner

    Parameters
    ----------
    ima : 3D array of doubles
        input image
    x : integer
        x coordinate of the center voxel
    y : integer
        y coordinate of the center voxel
    z : integer
        z coordinate of the center voxel
    average : 3D array of doubles
        the image where averages are stored
    weight : double
        weight for the weighted averaging
    """

    cdef int a, b, c, x_pos, y_pos, z_pos
    cdef int is_outside
    cdef int neighborhoodsize = average.shape[0] // 2
    for a in range(average.shape[0]):
        for b in range(average.shape[1]):
            for c in range(average.shape[2]):
                x_pos = x + a - neighborhoodsize
                y_pos = y + b - neighborhoodsize
                z_pos = z + c - neighborhoodsize
                is_outside = 0
                if x_pos < 0 or x_pos >= ima.shape[1]:
                    is_outside = 1
                if y_pos < 0 or y_pos >= ima.shape[0]:
                    is_outside = 1
                if z_pos < 0 or z_pos >= ima.shape[2]:
                    is_outside = 1
                if is_outside == 1:
                    average[a, b, c] += weight * (ima[y, x, z]**2)
                else:
                    average[a, b, c] += weight * (ima[y_pos, x_pos, z_pos]**2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _value_block(double[:, :, :] estimate, double[:, :, :] Label, int x, int y,
                       int z, double[:, :, :] average, double global_sum, double hh, int rician_int) noexcept nogil:

    """
    Computes the final estimate of the denoised image

    Parameters
    ----------
    estimate : 3D array of doubles
        The denoised estimate array
    Label : 3D array of doubles
        The label map for block wise weighted averaging
    x : integer
        x coordinate of the center voxel
    y : integer
        y coordinate of the center voxel
    z : integer
        z coordinate of the center voxel
    average : 3D array of doubles
        weighted average image
    global_sum : double
        total weight sum
    hh : double
        weight parameter
    rician_int : integer
        0 or 1 as per the boolean value
    """

    cdef int is_outside, a, b, c, x_pos, y_pos, z_pos, count = 0
    cdef double value = 0.0
    cdef double denoised_value = 0.0
    cdef double label = 0.0
    cdef int neighborhoodsize = average.shape[0] // 2
    for a in range(average.shape[0]):
        for b in range(average.shape[1]):
            for c in range(average.shape[2]):
                is_outside = 0
                x_pos = x + a - neighborhoodsize
                y_pos = y + b - neighborhoodsize
                z_pos = z + c - neighborhoodsize
                if x_pos < 0 or x_pos >= estimate.shape[1]:
                    is_outside = 1
                if y_pos < 0 or y_pos >= estimate.shape[0]:
                    is_outside = 1
                if z_pos < 0 or z_pos >= estimate.shape[2]:
                    is_outside = 1
                if is_outside == 0:
                    value = estimate[y_pos, x_pos, z_pos]
                    if rician_int:
                        denoised_value = (average[a, b, c] / global_sum) - hh
                    else:
                        denoised_value = (average[a, b, c] / global_sum)
                    if denoised_value > 0:
                        denoised_value = sqrt(denoised_value)
                    else:
                        denoised_value = 0.0
                    value += denoised_value
                    label = Label[y_pos, x_pos, z_pos]
                    estimate[y_pos, x_pos, z_pos] = value
                    Label[y_pos, x_pos, z_pos] = label + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _distance(double[:, :, :] image, int x, int y, int z,
                      int nx, int ny, int nz, int block_radius) nogil:
    """
    Computes the distance between two square subpatches of image located at
    p and q, respectively. If the centered squares lie beyond the boundaries
    of image, they are mirrored.

    Parameters
    ----------
    image : 3D array of doubles
        the image whose voxels are taken
    x : integer
        x coordinate of first patch's center
    y : integer
        y coordinate of first patch's center
    z : integer
        z coordinate of first patch's center
    nx : integer
        nx coordinate of second patch's center
    ny : integer
        ny coordinate of second patch's center
    nz : integer
        nz coordinate of second patch's center
    block_radius : integer
        block radius for which the distance is computed for
    """

    cdef double acu, distancetotal
    cdef int i, j, k, ni1, nj1, ni2, nj2, nk1, nk2
    cdef int sx = image.shape[1], sy = image.shape[0], sz = image.shape[2]
    acu = 0
    distancetotal = 0
    for i in range(-block_radius, block_radius + 1):
        for j in range(-block_radius, block_radius + 1):
            for k in range(-block_radius, block_radius + 1):
                ni1 = x + i
                nj1 = y + j
                nk1 = z + k
                ni2 = nx + i
                nj2 = ny + j
                nk2 = nz + k
                if ni1 < 0:
                    ni1 = -ni1
                if nj1 < 0:
                    nj1 = -nj1
                if ni2 < 0:
                    ni2 = -ni2
                if nj2 < 0:
                    nj2 = -nj2
                if nk1 < 0:
                    nk1 = -nk1
                if nk2 < 0:
                    nk2 = -nk2
                if ni1 >= sx:
                    ni1 = 2 * sx - ni1 - 1
                if nj1 >= sy:
                    nj1 = 2 * sy - nj1 - 1
                if nk1 >= sz:
                    nk1 = 2 * sz - nk1 - 1
                if ni2 >= sx:
                    ni2 = 2 * sx - ni2 - 1
                if nj2 >= sy:
                    nj2 = 2 * sy - nj2 - 1
                if nk2 >= sz:
                    nk2 = 2 * sz - nk2 - 1
                distancetotal += (image[nj1, ni1, nk1] -
                                  image[nj2, ni2, nk2])**2
                acu = acu + 1
    return distancetotal / acu


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _local_mean(double[:, :, :]ima, int x, int y, int z) nogil:
    """
    local mean of a 3x3x3 patch centered at x,y,z
    """
    cdef int dims0 = ima.shape[0]
    cdef int dims1 = ima.shape[1]
    cdef int dims2 = ima.shape[2]
    cdef double ss = 0
    cdef int px, py, pz, dx, dy, dz, nx, ny, nz
    for px in range(x - 1, x + 2):
        for py in range(y - 1, y + 2):
            for pz in range(z - 1, z + 2):
                px = (-px if px < 0 else (2 * dims0 -
                                          px - 1 if px >= dims0 else px))
                py = (-py if py < 0 else (2 * dims1 -
                                          py - 1 if py >= dims1 else py))
                pz = (-pz if pz < 0 else (2 * dims2 -
                                          pz - 1 if pz >= dims2 else pz))
                ss += ima[px, py, pz]

    return ss / 27.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _local_variance(double[:, :, :] ima, double mean, int x, int y, int z) nogil:
    """
    local variance of a 3x3x3 patch centered at x,y,z
    """
    dims0 = ima.shape[0]
    dims1 = ima.shape[1]
    dims2 = ima.shape[2]
    cdef int cnt = 0
    cdef double ss = 0
    cdef int dx, dy, dz, nx, ny, nz
    for px in range(x - 1, x + 2):
        for py in range(y - 1, y + 2):
            for pz in range(z - 1, z + 2):
                if ((px >= 0 and py >= 0 and pz > 0) and
                        (px < dims0 and py < dims1 and pz < dims2)):
                    ss += (ima[px, py, pz] - mean) * (ima[px, py, pz] - mean)
                    cnt += 1
    return ss / (cnt - 1)

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def nlmeans_block(double[:, :, :]image, double[:, :, :] mask, int patch_radius, int block_radius, double h, int rician):
    """Non-Local Means Denoising Using Blockwise Averaging.

    See :footcite:p:`Coupe2008` and :footcite:p:`Coupe2012` for further details
    about the method.

    Parameters
    ----------
    image : 3D array of doubles
        the input image, corrupted with rician noise
    mask : 3D array of doubles
        the input mask
    patch_radius :  int
        similar patches in the non-local means are searched for locally,
        inside a cube of side 2*v+1 centered at each voxel of interest.
    block_radius :  int
        the size of the block to be used (2*f+1)x(2*f+1)x(2*f+1) in the
        blockwise non-local means implementation (the Coupe's proposal).
    h :  double
        the estimated amount of rician noise in the input image: in P.
        Coupe et al. the rician noise was simulated as
        sqrt((f+x)^2 + (y)^2) where f is the pixel value and x and y are
        independent realizations of a random variable with Normal
        distribution, with mean=0 and standard deviation=h
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.

    Returns
    -------
    fima: 3D double array
        the denoised output which has the same shape as input image.

    References
    ----------
    .. footbibliography::

    """

    cdef int[:] dims = cvarray((3,), itemsize=sizeof(int), format="i")
    dims[0] = image.shape[0]
    dims[1] = image.shape[1]
    dims[2] = image.shape[2]
    cdef double hh = 2 * h * h
    cdef int Ndims = (2 * block_radius + 1)**3
    cdef int nvox = dims[0] * dims[1] * dims[2]
    cdef double[:, :, :] average = np.zeros((2 * block_radius + 1, 2 * block_radius + 1, 2 * block_radius + 1), dtype=np.float64)
    cdef double[:, :, :] fima = np.zeros_like(image)
    cdef double[:, :, :] means = np.zeros_like(image)
    cdef double[:, :, :] variances = np.zeros_like(image)
    cdef double[:, :, :] Estimate = np.zeros_like(image)
    cdef double[:, :, :] Label = np.zeros_like(image)
    cdef cnp.npy_intp i, j, k, ni, nj, nk
    cdef double t1, t2
    cdef double epsilon = 0.00001
    cdef double mu1 = 0.95
    cdef double var1 = 0.5 + 1e-7
    cdef double d
    cdef double totalWeight, wmax, w

    with nogil:
        for k in range(dims[2]):
            for i in range(dims[1]):
                for j in range(dims[0]):
                    means[j, i, k] = _local_mean(image, j, i, k)
                    variances[j, i, k] = _local_variance(
                        image, means[j, i, k], j, i, k)
        for k in range(0, dims[2], 2):
            for i in range(0, dims[1], 2):
                for j in range(0, dims[0], 2):
                    with gil:
                        average[...] = 0
                    totalWeight = 0
                    if (means[j, i, k] <= epsilon) or (
                            variances[j, i, k] <= epsilon):
                        wmax = 1.0
                        _average_block(image, i, j, k, average, wmax)
                        totalWeight += wmax
                        _value_block(Estimate, Label, i, j, k,
                                     average, totalWeight, hh, rician)
                    else:
                        wmax = 0
                        for nk in range(k - patch_radius, k + patch_radius + 1):
                            for ni in range(i - patch_radius, i + patch_radius + 1):
                                for nj in range(j - patch_radius, j + patch_radius + 1):
                                    if ni == i and nj == j and nk == k:
                                        continue
                                    if ni < 0 or nj < 0 or nk < 0 or nj >= dims[0] or ni >= dims[1] or nk >= dims[2]:
                                        continue
                                    if ((means[nj, ni, nk] <= epsilon) or (
                                            variances[nj, ni, nk] <= epsilon)):
                                        continue
                                    t1 = (means[j, i, k]) / (means[nj, ni, nk])
                                    t2 = (variances[j, i, k]) / \
                                        (variances[nj, ni, nk])
                                    if mu1 < t1 < (1 / mu1) and var1 < t2 < (1 / var1):
                                        d = _distance(
                                            image, i, j, k, ni, nj, nk, block_radius)
                                        w = exp(-d / (h * h))
                                        if w > wmax:
                                            wmax = w
                                        _average_block(
                                            image, ni, nj, nk, average, w)
                                        totalWeight += w

                        if totalWeight != 0.0:
                            _value_block(Estimate, Label, i, j, k,
                                         average, totalWeight, hh, rician)

        for k in range(0, dims[2]):
            for i in range(0, dims[1]):
                for j in range(0, dims[0]):

                    if mask[j, i, k] == 0:
                        fima[j, i, k] = 0

                    else:
                        if Label[j, i, k] == 0.0:
                            fima[j, i, k] = image[j, i, k]
                        else:
                            fima[j, i, k] = Estimate[j, i, k] / Label[j, i, k]

    return fima
