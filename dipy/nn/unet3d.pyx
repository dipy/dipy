# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: wraparound=False
"""CPU inference of the Synb0 3D UNet, without any deep learning framework.

Activations are stored channel-last in volumes zero padded by one voxel on
each side, shape (D + 2, H + 2, W + 2, C). In the flattened padded volume,
the 27 taps of a 3x3x3 convolution are fixed offsets, so the convolution is a
sum of 27 matrix products over contiguous rows, computed with BLAS ``sgemm``.
The rows mapped to the padding receive meaningless values, which the
normalization zeroes afterwards.
"""

import numpy as np

from cython.parallel cimport prange
from libc.math cimport sqrt
from scipy.linalg.cython_blas cimport sgemm

from dipy.utils.omp import determine_num_threads


cdef float INSTANCE_NORM_EPS = <float>1e-5
cdef float LEAKY_RELU_SLOPE = <float>0.01
cdef char NO_TRANSPOSE = b"N"
# Number of times the UNet halves the volume size.
DEPTH = 3


cdef void _conv3x3(float[:, :, :, ::1] src, float[:, :, ::1] kernel,
                   float[:, :, :, ::1] dst, bint accumulate) noexcept nogil:
    """Zero-padded 3x3x3 convolution, added to ``dst`` if ``accumulate``."""
    cdef:
        int hp = <int>src.shape[1]
        int wp = <int>src.shape[2]
        int n_in = <int>src.shape[3]
        int n_out = <int>dst.shape[3]
        int n_rows = (<int>src.shape[0] - 3) * hp * wp + (hp - 3) * wp + wp - 2
        int tap, offset
        float one = 1, beta
        float *out = &dst[0, 0, 0, 0] + (hp * wp + wp + 1) * n_out
        float *inp = &src[0, 0, 0, 0]

    for tap in range(27):
        offset = (tap // 9) * hp * wp + (tap // 3 % 3) * wp + tap % 3
        beta = 1 if accumulate or tap > 0 else 0
        sgemm(&NO_TRANSPOSE, &NO_TRANSPOSE, &n_out, &n_rows, &n_in, &one,
              &kernel[tap, 0, 0], &n_out, inp + offset * n_in, &n_in, &beta,
              out, &n_out)


cdef void _conv1x1(float[:, :, :, ::1] src, float[:, ::1] kernel,
                   float[:, :, :, ::1] dst) noexcept nogil:
    """1x1x1 convolution over the whole padded volume."""
    cdef:
        int n_in = <int>src.shape[3]
        int n_out = <int>dst.shape[3]
        int n_rows = <int>(src.shape[0] * src.shape[1] * src.shape[2])
        float one = 1, zero = 0

    sgemm(&NO_TRANSPOSE, &NO_TRANSPOSE, &n_out, &n_rows, &n_in, &one,
          &kernel[0, 0], &n_out, &src[0, 0, 0, 0], &n_in, &zero,
          &dst[0, 0, 0, 0], &n_out)


cdef void _upconv2x2(float[:, :, :, ::1] src, float[:, ::1] kernel,
                     float[:, ::1] buffer, float[:, :, :, ::1] dst,
                     int n_threads) noexcept nogil:
    """2x2x2 transposed convolution with stride 2, one input plane at a time.

    ``kernel`` is indexed (input channel, (a, b, c, output channel)), where
    (a, b, c) is the position of the output voxel in its 2x2x2 block.
    """
    cdef:
        int hp = <int>src.shape[1]
        int wp = <int>src.shape[2]
        int n_in = <int>src.shape[3]
        int n_out = <int>dst.shape[3]
        int n_cols = 8 * n_out
        int n_rows = hp * wp
        Py_ssize_t z, y, x, block, c, row
        float one = 1, zero = 0

    for z in range(src.shape[0] - 2):
        sgemm(&NO_TRANSPOSE, &NO_TRANSPOSE, &n_cols, &n_rows, &n_in, &one,
              &kernel[0, 0], &n_cols, &src[z + 1, 0, 0, 0], &n_in, &zero,
              &buffer[0, 0], &n_cols)
        for y in prange(hp - 2, num_threads=n_threads, schedule="static"):
            for x in range(wp - 2):
                row = (y + 1) * wp + x + 1
                for block in range(8):
                    for c in range(n_out):
                        dst[2 * z + 1 + block // 4, 2 * y + 1 + block // 2 % 2,
                            2 * x + 1 + block % 2, c] = buffer[row,
                                                               block * n_out + c]


cdef void _norm_leaky_relu(float[:, :, :, ::1] buf, double[:, ::1] sums,
                           double[:, ::1] stats, int n_threads) noexcept nogil:
    """Instance normalization then leaky ReLU, zeroing the padding.

    ``sums`` holds one row of partial sums per plane, ``stats`` receives the
    mean and the inverse standard deviation of each channel.
    """
    cdef:
        Py_ssize_t nz = buf.shape[0] - 2
        Py_ssize_t ny = buf.shape[1] - 2
        Py_ssize_t nx = buf.shape[2] - 2
        Py_ssize_t nc = buf.shape[3]
        Py_ssize_t z, y, x, c, i
        double n_voxels = <double>(nz * ny * nx)
        double diff
        float value

    for z in prange(nz, num_threads=n_threads, schedule="static"):
        for c in range(nc):
            sums[z, c] = 0
        for y in range(1, ny + 1):
            for x in range(1, nx + 1):
                for c in range(nc):
                    sums[z, c] += buf[z + 1, y, x, c]
    for c in range(nc):
        stats[0, c] = 0
        for i in range(nz):
            stats[0, c] += sums[i, c]
        stats[0, c] /= n_voxels

    # Variance from the deviations to the mean, more accurate than E[x^2]-m^2.
    for z in prange(nz, num_threads=n_threads, schedule="static"):
        for c in range(nc):
            sums[z, c] = 0
        for y in range(1, ny + 1):
            for x in range(1, nx + 1):
                for c in range(nc):
                    diff = buf[z + 1, y, x, c] - stats[0, c]
                    sums[z, c] += diff * diff
    for c in range(nc):
        stats[1, c] = 0
        for i in range(nz):
            stats[1, c] += sums[i, c]
        stats[1, c] = 1 / sqrt(stats[1, c] / n_voxels + INSTANCE_NORM_EPS)

    for z in prange(nz + 2, num_threads=n_threads, schedule="static"):
        for y in range(ny + 2):
            for x in range(nx + 2):
                if z == 0 or z == nz + 1 or y == 0 or y == ny + 1 or x == 0 \
                        or x == nx + 1:
                    for c in range(nc):
                        buf[z, y, x, c] = 0
                else:
                    for c in range(nc):
                        value = <float>((buf[z, y, x, c] - stats[0, c])
                                        * stats[1, c])
                        buf[z, y, x, c] = value if value > 0 \
                            else LEAKY_RELU_SLOPE * value


cdef void _max_pool(float[:, :, :, ::1] src, float[:, :, :, ::1] dst,
                    int n_threads) noexcept nogil:
    """2x2x2 max pooling with stride 2."""
    cdef:
        Py_ssize_t z, y, x, c, block
        float value, best

    for z in prange(dst.shape[0] - 2, num_threads=n_threads, schedule="static"):
        for y in range(dst.shape[1] - 2):
            for x in range(dst.shape[2] - 2):
                for c in range(dst.shape[3]):
                    best = src[2 * z + 1, 2 * y + 1, 2 * x + 1, c]
                    for block in range(1, 8):
                        value = src[2 * z + 1 + block // 4,
                                    2 * y + 1 + block // 2 % 2,
                                    2 * x + 1 + block % 2, c]
                        if value > best:
                            best = value
                    dst[z + 1, y + 1, x + 1, c] = best


def _padded(shape, channels):
    """Allocate a zero padded channel-last activation volume.

    Parameters
    ----------
    shape : tuple of int
        Shape (D, H, W) of the volume, without padding.
    channels : int
        Number of channels.

    Returns
    -------
    np.ndarray (D + 2, H + 2, W + 2, channels)
        Zeros, as float32.
    """
    return np.zeros(tuple(s + 2 for s in shape) + (channels,), dtype=np.float32)


def _conv_kernel(weight, *, transposed):
    """Pack a 3x3x3 PyTorch kernel as (tap, input channel, output channel).

    Parameters
    ----------
    weight : np.ndarray
        ``Conv3d`` weight (out, in, 3, 3, 3), or ``ConvTranspose3d`` weight
        (in, out, 3, 3, 3) with stride 1 and padding 1.
    transposed : bool
        Whether ``weight`` comes from a ``ConvTranspose3d``. Such a layer is
        a convolution with the kernel flipped and its channel axes swapped.

    Returns
    -------
    np.ndarray (27, in, out)
        Packed kernel.
    """
    if transposed:
        weight = weight[:, :, ::-1, ::-1, ::-1].transpose(2, 3, 4, 0, 1)
    else:
        weight = weight.transpose(2, 3, 4, 1, 0)
    return np.ascontiguousarray(weight.reshape((27,) + weight.shape[3:]),
                                dtype=np.float32)


def _upconv_kernel(weight):
    """Pack a 2x2x2 ``ConvTranspose3d`` kernel as (in, (a, b, c, out)).

    Parameters
    ----------
    weight : np.ndarray (in, out, 2, 2, 2)
        PyTorch weight.

    Returns
    -------
    np.ndarray (in, 8 * out)
        Packed kernel.
    """
    return np.ascontiguousarray(
        weight.transpose(0, 2, 3, 4, 1).reshape(weight.shape[0], -1),
        dtype=np.float32,
    )


def _weight(state, name, shape):
    """Fetch a weight from a state dict and check its shape.

    Parameters
    ----------
    state : dict
        Parameter names mapped to arrays.
    name : str
        Name of the parameter.
    shape : tuple of int
        Expected shape.

    Returns
    -------
    np.ndarray
        The weight, as float32.
    """
    try:
        weight = np.asarray(state[name], dtype=np.float32)
    except KeyError as e:
        raise ValueError(f"Missing UNet3D weight {name}") from e
    if weight.shape != tuple(shape):
        raise ValueError(
            f"UNet3D weight {name} should have shape {tuple(shape)}, "
            f"got {weight.shape}"
        )
    return weight


def pack_unet3d_weights(state):
    """Pack the PyTorch weights of the Synb0 UNet3D for ``unet3d_forward_cpu``.

    Parameters
    ----------
    state : dict
        State dict of ``dipy.nn.synb0.UNet3D``, parameter names mapped to
        arrays.

    Returns
    -------
    dict
        Packed kernels, keyed by layer name.
    """
    widths = (32, 64, 64, 128, 128, 256, 256, 512)
    expected = {f"ec{i}.0.weight" for i in range(len(widths))}
    expected |= {f"dc{i}.0.weight" for i in range(10)}
    expected |= {"el.weight", "el.bias", "dl.weight", "dl.bias"}
    if set(state) != expected:
        raise ValueError("The state dict does not hold UNet3D weights")
    n_in = np.shape(state["ec0.0.weight"])[1]
    n_out = np.shape(state["dl.bias"])[0]
    packed = {}
    previous = n_in
    for i, width in enumerate(widths):
        name = f"ec{i}.0.weight"
        packed[f"ec{i}"] = _conv_kernel(
            _weight(state, name, (width, previous, 3, 3, 3)), transposed=False
        )
        previous = width
    packed["el"] = np.ascontiguousarray(
        _weight(state, "el.weight", (512, 512, 1, 1, 1))[:, :, 0, 0, 0].T
    )
    packed["el_bias"] = _weight(state, "el.bias", (512,))

    # Each level upsamples, then convolves the upsampled activations
    # concatenated with the skip connection, then convolves again.
    for up, conv, skip, width, upsampled in (
        (9, 8, 256, 256, 512),
        (6, 5, 128, 128, 256),
        (3, 2, 64, 64, 128),
    ):
        packed[f"dc{up}"] = _upconv_kernel(
            _weight(state, f"dc{up}.0.weight", (upsampled, upsampled, 2, 2, 2))
        )
        kernel = _conv_kernel(
            _weight(state, f"dc{conv}.0.weight",
                    (upsampled + skip, width, 3, 3, 3)),
            transposed=True,
        )
        packed[f"dc{conv}"] = np.ascontiguousarray(kernel[:, :upsampled])
        packed[f"dc{conv}_skip"] = np.ascontiguousarray(kernel[:, upsampled:])
        packed[f"dc{conv - 1}"] = _conv_kernel(
            _weight(state, f"dc{conv - 1}.0.weight", (width, width, 3, 3, 3)),
            transposed=True,
        )
    packed["dc0"] = np.ascontiguousarray(
        _weight(state, "dc0.0.weight", (64, n_out, 1, 1, 1))[:, :, 0, 0, 0]
    )
    packed["dl"] = np.ascontiguousarray(
        _weight(state, "dl.weight", (n_out, n_out, 1, 1, 1))[:, :, 0, 0, 0]
    )
    packed["dl_bias"] = _weight(state, "dl.bias", (n_out,))
    return packed


def _block(src, kernel, n_out, n_threads, *, skip=None, skip_kernel=None):
    """3x3x3 convolution, instance normalization and leaky ReLU.

    Parameters
    ----------
    src : np.ndarray (D + 2, H + 2, W + 2, C)
        Padded input.
    kernel : np.ndarray (27, C, n_out)
        Packed kernel.
    n_out : int
        Number of output channels.
    n_threads : int
        Number of OpenMP threads.
    skip : np.ndarray (D + 2, H + 2, W + 2, S), optional
        Skip connection, concatenated after ``src`` along the channels.
    skip_kernel : np.ndarray (27, S, n_out), optional
        Packed kernel of the skip connection channels.

    Returns
    -------
    np.ndarray (D + 2, H + 2, W + 2, n_out)
        Padded output.
    """
    cdef:
        float[:, :, :, ::1] c_src = src
        float[:, :, :, ::1] c_dst
        float[:, :, :, ::1] c_skip
        float[:, :, ::1] c_kernel = kernel
        float[:, :, ::1] c_skip_kernel
        double[:, ::1] sums
        double[:, ::1] stats
        int threads = n_threads

    shape = tuple(s - 2 for s in src.shape[:3])
    dst = _padded(shape, n_out)
    c_dst = dst
    sums = np.empty((shape[0], n_out))
    stats = np.empty((2, n_out))
    with nogil:
        _conv3x3(c_src, c_kernel, c_dst, False)
    if skip is not None:
        c_skip = skip
        c_skip_kernel = skip_kernel
        with nogil:
            _conv3x3(c_skip, c_skip_kernel, c_dst, True)
    with nogil:
        _norm_leaky_relu(c_dst, sums, stats, threads)
    return dst


def _pool(src, n_threads):
    """2x2x2 max pooling of a padded volume.

    Parameters
    ----------
    src : np.ndarray (D + 2, H + 2, W + 2, C)
        Padded input.
    n_threads : int
        Number of OpenMP threads.

    Returns
    -------
    np.ndarray (D // 2 + 2, H // 2 + 2, W // 2 + 2, C)
        Padded output.
    """
    cdef:
        float[:, :, :, ::1] c_src = src
        float[:, :, :, ::1] c_dst
        int threads = n_threads

    dst = _padded(tuple((s - 2) // 2 for s in src.shape[:3]), src.shape[3])
    c_dst = dst
    with nogil:
        _max_pool(c_src, c_dst, threads)
    return dst


def _upsample(src, kernel, n_threads):
    """2x2x2 transposed convolution, instance normalization and leaky ReLU.

    Parameters
    ----------
    src : np.ndarray (D + 2, H + 2, W + 2, C)
        Padded input.
    kernel : np.ndarray (C, 8 * n_out)
        Packed kernel.
    n_threads : int
        Number of OpenMP threads.

    Returns
    -------
    np.ndarray (2 * D + 2, 2 * H + 2, 2 * W + 2, n_out)
        Padded output.
    """
    cdef:
        float[:, :, :, ::1] c_src = src
        float[:, :, :, ::1] c_dst
        float[:, ::1] c_kernel = kernel
        float[:, ::1] buffer
        double[:, ::1] sums
        double[:, ::1] stats
        int threads = n_threads

    n_out = kernel.shape[1] // 8
    shape = tuple(2 * (s - 2) for s in src.shape[:3])
    dst = _padded(shape, n_out)
    c_dst = dst
    buffer = np.empty((src.shape[1] * src.shape[2], kernel.shape[1]),
                      dtype=np.float32)
    sums = np.empty((shape[0], n_out))
    stats = np.empty((2, n_out))
    with nogil:
        _upconv2x2(c_src, c_kernel, buffer, c_dst, threads)
        _norm_leaky_relu(c_dst, sums, stats, threads)
    return dst


def _normalize(x, n_threads):
    """Instance normalization and leaky ReLU of a padded volume, in place.

    Parameters
    ----------
    x : np.ndarray (D + 2, H + 2, W + 2, C)
        Padded volume.
    n_threads : int
        Number of OpenMP threads.
    """
    cdef:
        float[:, :, :, ::1] c_x = x
        double[:, ::1] sums = np.empty((x.shape[0] - 2, x.shape[3]))
        double[:, ::1] stats = np.empty((2, x.shape[3]))
        int threads = n_threads

    with nogil:
        _norm_leaky_relu(c_x, sums, stats, threads)


def _pointwise(src, kernel):
    """1x1x1 convolution of a padded volume, without bias.

    Parameters
    ----------
    src : np.ndarray (D + 2, H + 2, W + 2, C)
        Padded input.
    kernel : np.ndarray (C, n_out)
        Kernel.

    Returns
    -------
    np.ndarray (D + 2, H + 2, W + 2, n_out)
        Padded output.
    """
    cdef:
        float[:, :, :, ::1] c_src = src
        float[:, :, :, ::1] c_dst
        float[:, ::1] c_kernel = kernel

    dst = np.empty(src.shape[:3] + (kernel.shape[1],), dtype=np.float32)
    c_dst = dst
    with nogil:
        _conv1x1(c_src, c_kernel, c_dst)
    return dst


def unet3d_forward_cpu(volume, weights, *, num_threads=None):
    """Run the Synb0 3D UNet on the CPU.

    Same network as ``dipy.nn.synb0.UNet3D`` :footcite:p:`Schilling2019`:
    three levels of two 3x3x3 convolutions and a 2x2x2 max pooling, two
    3x3x3 convolutions and a 1x1x1 convolution at the bottom, then three
    levels of a 2x2x2 transposed convolution with stride 2 and two 3x3x3
    transposed convolutions taking the skip connection, and two 1x1x1
    transposed convolutions. Every convolution of the encoder and decoder
    blocks is followed by a parameter-free instance normalization and a leaky
    ReLU of slope 0.01.

    Parameters
    ----------
    volume : np.ndarray (n_in, D, H, W)
        Input channels. D, H and W should be multiples of 8.
    weights : dict
        PyTorch state dict of the network, or the output of
        ``pack_unet3d_weights``.
    num_threads : int, optional
        Number of threads of the OpenMP loops. If None, the value of the
        ``OMP_NUM_THREADS`` environment variable is used if set, otherwise
        all available cores. If < 0, the maximal number of threads minus
        $|num_threads + 1|$ is used (enter -1 to use as many threads as
        possible). 0 raises an error. The matrix products use the threads of
        the BLAS library.

    Returns
    -------
    np.ndarray (n_out, D, H, W)
        Network output, as float32.

    References
    ----------
    .. footbibliography::
    """
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 4:
        raise ValueError(f"volume should be 4D (n_in, D, H, W), got {volume.ndim}D")
    factor = 2 ** DEPTH
    if any(s % factor for s in volume.shape[1:]):
        raise ValueError(
            f"The spatial shape of volume should be a multiple of {factor}, "
            f"got {volume.shape[1:]}"
        )
    if "ec0" not in weights:
        weights = pack_unet3d_weights(weights)
    if weights["ec0"].shape[1] != volume.shape[0]:
        raise ValueError(
            f"volume should have {weights['ec0'].shape[1]} channels, "
            f"got {volume.shape[0]}"
        )
    threads = determine_num_threads(num_threads)

    x = _padded(volume.shape[1:], volume.shape[0])
    x[1:-1, 1:-1, 1:-1] = np.moveaxis(volume, 0, -1)

    skips = []
    for level in range(DEPTH):
        kernel = weights[f"ec{2 * level}"]
        x = _block(x, kernel, kernel.shape[2], threads)
        kernel = weights[f"ec{2 * level + 1}"]
        x = _block(x, kernel, kernel.shape[2], threads)
        skips.append(x)
        x = _pool(x, threads)
    for i in (6, 7):
        x = _block(x, weights[f"ec{i}"], weights[f"ec{i}"].shape[2], threads)
    x = _pointwise(x, weights["el"])
    x += weights["el_bias"]

    for up, conv in ((9, 8), (6, 5), (3, 2)):
        x = _upsample(x, weights[f"dc{up}"], threads)
        kernel = weights[f"dc{conv}"]
        x = _block(x, kernel, kernel.shape[2], threads, skip=skips.pop(),
                   skip_kernel=weights[f"dc{conv}_skip"])
        kernel = weights[f"dc{conv - 1}"]
        x = _block(x, kernel, kernel.shape[2], threads)

    x = _pointwise(x, weights["dc0"])
    _normalize(x, threads)
    out = x[1:-1, 1:-1, 1:-1] @ weights["dl"] + weights["dl_bias"]
    return np.ascontiguousarray(np.moveaxis(out, -1, 0))
