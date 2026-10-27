# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: wraparound=False
"""CPU inference of MeshNet models, without any deep learning framework."""

import numpy as np

from dipy.utils.omp import determine_num_threads


cdef extern from "meshnet_cpu.h" nogil:
    int MESHNET_STRIDE
    int MESHNET_MAX_CLASSES
    int meshnet_forward(const float *volume, float *logits, int nz, int ny,
                        int nx, const float *first_kernel,
                        const float *hidden_kernels, const int *dilations,
                        int n_layers, int channels, const float *classifier,
                        const float *bias, int n_classes, int n_threads)
    const char *meshnet_kernels()
    void meshnet_allow_simd(int allow)


def meshnet_cpu_kernels():
    """Name the instruction set used by ``meshnet_forward_cpu``.

    On x86 CPUs supporting AVX2 and FMA, a copy of the kernels built for
    these instructions is selected at run time, whatever the instruction set
    DIPY was compiled for.

    Returns
    -------
    str
        ``"avx2"`` or ``"baseline"``.
    """
    return meshnet_kernels().decode()


def _set_simd_allowed(*, allow):
    """Allow or forbid the AVX2 kernels, to compare both in tests.

    Parameters
    ----------
    allow : bool
        Whether the AVX2 kernels may be selected.
    """
    meshnet_allow_simd(1 if allow else 0)


def _pack_kernels(kernels, channels):
    """Pack PyTorch-layout kernels into the channel-last layout of the C code.

    Parameters
    ----------
    kernels : sequence of np.ndarray (channels, in_channels, 3, 3, 3)
        Convolution kernels, the first one with a single input channel.
    channels : int
        Number of output channels of every kernel.

    Returns
    -------
    first : np.ndarray (27, MESHNET_STRIDE)
        First kernel, zero padded.
    hidden : np.ndarray (n_hidden, 27, MESHNET_STRIDE, MESHNET_STRIDE)
        Remaining kernels, zero padded on both channel axes.
    """
    first = np.zeros((27, MESHNET_STRIDE), dtype=np.float32)
    first[:, :channels] = kernels[0][:, 0].reshape(channels, 27).T
    hidden = np.zeros(
        (max(1, len(kernels) - 1), 27, MESHNET_STRIDE, MESHNET_STRIDE),
        dtype=np.float32,
    )
    for layer, kernel in enumerate(kernels[1:]):
        hidden[layer, :, :channels, :channels] = kernel.transpose(
            2, 3, 4, 1, 0
        ).reshape(27, channels, channels)
    return first, hidden


def meshnet_forward_cpu(volume, kernels, dilations, classifier, bias, *,
                        num_threads=None):
    """Compute the class logits of a MeshNet model on the CPU.

    MeshNet :footcite:p:`Fedorov2017` stacks zero-padded dilated 3x3x3
    convolutions, each followed by a parameter-free group normalization with
    one group per channel and a GELU activation (tanh approximation), then a
    1x1x1 convolution giving the class logits.

    Parameters
    ----------
    volume : np.ndarray (X, Y, Z)
        Input volume, with a single channel.
    kernels : sequence of np.ndarray (channels, in_channels, 3, 3, 3)
        Weights of the dilated convolutions, in the PyTorch ``Conv3d``
        layout. The first kernel has a single input channel, the following
        ones have ``channels`` input channels.
    dilations : sequence of int
        Dilation of each convolution.
    classifier : np.ndarray (n_classes, channels)
        Weights of the final 1x1x1 convolution.
    bias : np.ndarray (n_classes,)
        Bias of the final 1x1x1 convolution.
    num_threads : int, optional
        Number of threads to use. If None, the value of the
        ``OMP_NUM_THREADS`` environment variable is used if set, otherwise
        all available cores. If < 0, the maximal number of threads minus
        $|num_threads + 1|$ is used (enter -1 to use as many threads as
        possible). 0 raises an error.

    Returns
    -------
    np.ndarray (n_classes, X, Y, Z)
        Class logits, as float32.

    References
    ----------
    .. footbibliography::
    """
    cdef:
        float[:, :, ::1] c_volume
        float[:, :, :, ::1] c_logits
        float[:, ::1] c_first
        float[:, :, :, ::1] c_hidden
        int[::1] c_dilations
        float[:, ::1] c_classifier
        float[::1] c_bias
        int nz, ny, nx, n_layers, channels, n_classes, threads, status

    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"volume should be 3D, got {volume.ndim}D")
    kernels = [np.asarray(kernel, dtype=np.float32) for kernel in kernels]
    n_layers = len(kernels)
    if n_layers == 0 or len(dilations) != n_layers:
        raise ValueError("kernels and dilations should have the same length")
    channels = kernels[0].shape[0]
    if not 1 <= channels <= MESHNET_STRIDE:
        raise ValueError(f"channels should be between 1 and {MESHNET_STRIDE}")
    if kernels[0].shape != (channels, 1, 3, 3, 3):
        raise ValueError("the first kernel should have shape (channels, 1, 3, 3, 3)")
    for kernel in kernels[1:]:
        if kernel.shape != (channels, channels, 3, 3, 3):
            raise ValueError(
                "hidden kernels should have shape (channels, channels, 3, 3, 3)"
            )
    classifier = np.asarray(classifier, dtype=np.float32)
    n_classes = classifier.shape[0]
    if classifier.shape != (n_classes, channels) or not (
        1 <= n_classes <= MESHNET_MAX_CLASSES
    ):
        raise ValueError(
            f"classifier should have shape (n_classes, {channels}) with at most "
            f"{MESHNET_MAX_CLASSES} classes"
        )
    bias = np.ascontiguousarray(bias, dtype=np.float32)
    if bias.shape != (n_classes,):
        raise ValueError(f"bias should have shape ({n_classes},)")
    dilations = np.ascontiguousarray(dilations, dtype=np.intc)
    if np.any(dilations < 1):
        raise ValueError("dilations should be positive")

    first, hidden = _pack_kernels(kernels, channels)
    logits = np.empty((n_classes,) + volume.shape, dtype=np.float32)

    c_volume = np.ascontiguousarray(volume, dtype=np.float32)
    c_logits = logits
    c_first = first
    c_hidden = hidden
    c_dilations = dilations
    c_classifier = np.ascontiguousarray(classifier.T)
    c_bias = bias
    nz, ny, nx = volume.shape
    threads = determine_num_threads(num_threads)

    with nogil:
        status = meshnet_forward(
            &c_volume[0, 0, 0], &c_logits[0, 0, 0, 0], nz, ny, nx,
            &c_first[0, 0], &c_hidden[0, 0, 0, 0], &c_dilations[0], n_layers,
            channels, &c_classifier[0, 0], &c_bias[0], n_classes, threads,
        )
    if status == 2:
        raise MemoryError("Not enough memory for the MeshNet activations")
    if status != 0:
        raise ValueError("Invalid MeshNet arguments")
    return logits
