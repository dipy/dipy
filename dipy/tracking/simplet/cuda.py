"""
CUDA backend for :mod:`dipy.tracking.simplet.tracker`.
"""

from importlib.resources import files
import logging
from time import time

import numpy as np

from dipy.utils.optpkg import optional_package

logger = logging.getLogger("dipy")

_TRIP_MSG = (
    "CUDA tracking requires cuda-python, cuda-core and cuda-cccl. "
    'Install them with: pip install "dipy[cu12]" (or "dipy[cu13]")'
)
driver, have_driver, _ = optional_package("cuda.bindings.driver", trip_msg=_TRIP_MSG)
nvrtc, _, _ = optional_package("cuda.bindings.nvrtc", trip_msg=_TRIP_MSG)
runtime, have_runtime, _ = optional_package("cuda.bindings.runtime", trip_msg=_TRIP_MSG)
cccl, have_cccl, _ = optional_package("cuda.cccl", trip_msg=_TRIP_MSG)
core, have_core, _ = optional_package("cuda.core", trip_msg=_TRIP_MSG)
pathfinder, _, _ = optional_package("cuda.pathfinder", trip_msg=_TRIP_MSG)

have_cuda = have_driver and have_runtime and have_core and have_cccl

KERNEL_DTYPE = np.float32
INT32_NBYTES = np.dtype(np.int32).itemsize
THREADS_PER_STREAMLINE = 32
THREADS_PER_BLOCK = 64
STREAMLINES_PER_BLOCK = THREADS_PER_BLOCK // THREADS_PER_STREAMLINE
# symbol name in kernel.cu
KERNEL_NAME = f"genStreamlinesProb_k<{THREADS_PER_STREAMLINE},{STREAMLINES_PER_BLOCK}>"


def cuda_available():
    if not have_cuda:
        return False
    err, count = runtime.cudaGetDeviceCount()
    return err == runtime.cudaError_t.cudaSuccess and count > 0


def _check(result, *, hard_error=True):
    """Unwrap a cuda-python ``(err, *values)`` result, raising on error."""
    err = result[0]
    if err.value:
        if isinstance(err, driver.CUresult):
            _, name = driver.cuGetErrorName(err)
        elif isinstance(err, nvrtc.nvrtcResult):
            name = nvrtc.nvrtcGetErrorString(err)[1]
        else:
            name = runtime.cudaGetErrorString(err)[1]
        msg = f"CUDA error code={err.value} ({name})"
        if hard_error:
            raise RuntimeError(msg)
        logger.warning(msg)
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def _ceil_div(a, b):
    return (a + b - 1) // b


def _to_device(host):
    host = np.asarray(host, order="C")
    dev = _check(runtime.cudaMalloc(host.nbytes))
    _check(
        runtime.cudaMemcpy(
            dev,
            host.ctypes.data,
            host.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
    )
    return dev


def _allocate_texture_border(data):
    """
    Upload a 3D float32 volume as a linearly filtered texture whose
    out-of-bounds value is -1 (used by the kernel to detect leaving the image).
    """
    channel = _check(
        runtime.cudaCreateChannelDesc(
            32, 0, 0, 0, runtime.cudaChannelFormatKind.cudaChannelFormatKindFloat
        )
    )
    dim0, dim1, dim2 = data.shape
    extent = runtime.make_cudaExtent(dim2, dim1, dim0)
    array = _check(runtime.cudaMalloc3DArray(channel, extent, 0))

    copy = runtime.cudaMemcpy3DParms()
    copy.srcPtr = runtime.make_cudaPitchedPtr(data.ctypes.data, dim2 * 4, dim2, dim1)
    copy.dstArray = array
    copy.extent = extent
    copy.kind = runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
    _check(runtime.cudaMemcpy3D(copy))

    res = runtime.cudaResourceDesc()
    res.resType = runtime.cudaResourceType.cudaResourceTypeArray
    res.res.array.array = array

    tex = runtime.cudaTextureDesc()
    border = runtime.cudaTextureAddressMode.cudaAddressModeBorder
    tex.addressMode[0] = border
    tex.addressMode[1] = border
    tex.addressMode[2] = border
    tex.borderColor[0] = -1.0
    tex.borderColor[1] = -1.0
    tex.borderColor[2] = -1.0
    tex.filterMode = runtime.cudaTextureFilterMode.cudaFilterModeLinear
    tex.readMode = runtime.cudaTextureReadMode.cudaReadModeElementType
    tex.normalizedCoords = 0

    tex_obj = _check(runtime.cudaCreateTextureObject(res, tex, None))
    return tex_obj, array


def _compile(tracker_data):
    start = time()
    logger.info("Compiling CUDA simple tracker kernel...")
    dim_x, dim_y, dim_z, dim_t = tracker_data.shape
    # keys are the macro names used by kernel.cu
    macros = {
        "DIMX": dim_x,
        "DIMY": dim_y,
        "DIMZ": dim_z,
        "DIMT": dim_t,
        "STEP_SIZE": tracker_data.step_size,
        "MAX_ANGLE": tracker_data.max_angle,
        "TC_THRESHOLD": tracker_data.stop_threshold,
        "PMF_THRESHOLD_P": tracker_data.pmf_threshold,
        "MAX_SLINE_LEN": tracker_data.max_steps,
        "RNG_SEED": tracker_data.random_seed,
        "SPHERE_SYMM": 1 if tracker_data.is_symmetric else 0,
        "THR_X_SL": THREADS_PER_STREAMLINE,
        "THR_X_BL": THREADS_PER_BLOCK,
    }
    options = core.ProgramOptions(
        name="dipy_simplet",
        use_fast_math=True,
        std="c++17",
        define_macro=[f"{k}={v}" for k, v in macros.items()],
        include_path=[
            pathfinder.find_nvidia_header_directory("cudart"),
            pathfinder.find_nvidia_header_directory("curand"),
            cccl.get_include_paths().libcudacxx,
        ],
        ptxas_options=["-O3"],
    )
    # Compile once on the current device; all GPUs are assumed identical.
    dev = core.Device()
    dev.set_current()
    source = files("dipy.tracking.simplet").joinpath("kernel.cu").read_text()
    prog = core.Program(source, code_type="c++", options=options)
    module = prog.compile("cubin", name_expressions=(KERNEL_NAME,))
    logger.info("CUDA simple tracker kernel compiled in %.2f s", time() - start)
    return module.get_kernel(KERNEL_NAME)


def _launch(
    kernel,
    stream,
    device_data,
    seeds,
    streamline_offsets,
    peak_dirs,
    streamline_lengths_host,
    streamline_buffer_host,
):
    pmf_d, stop_map_tex, sphere_vertices_d = device_data
    n_seeds = len(seeds)
    n_streamlines = int(streamline_offsets[-1])
    if n_seeds == 0 or n_streamlines == 0:
        return

    seeds_d = _to_device(seeds)
    offsets_d = _to_device(streamline_offsets)
    dirs_d = _to_device(peak_dirs)
    seed_of_streamline_d = _check(runtime.cudaMalloc(INT32_NBYTES * n_streamlines))
    streamline_lengths_d = _check(runtime.cudaMalloc(INT32_NBYTES * n_streamlines))
    streamline_buffer_d = _check(runtime.cudaMalloc(streamline_buffer_host.nbytes))

    config = core.LaunchConfig(
        block=(THREADS_PER_STREAMLINE, STREAMLINES_PER_BLOCK, 1),
        grid=(_ceil_div(n_seeds, STREAMLINES_PER_BLOCK), 1, 1),
        shmem_size=0,
    )
    core.launch(
        stream,
        config,
        kernel,
        n_seeds,
        seeds_d,
        pmf_d,
        stop_map_tex.getPtr(),
        sphere_vertices_d,
        offsets_d,
        dirs_d,
        seed_of_streamline_d,
        streamline_lengths_d,
        streamline_buffer_d,
    )
    _check(runtime.cudaStreamSynchronize(stream))

    _check(
        runtime.cudaMemcpy(
            streamline_buffer_host.ctypes.data,
            streamline_buffer_d,
            streamline_buffer_host.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
    )
    _check(
        runtime.cudaMemcpy(
            streamline_lengths_host.ctypes.data,
            streamline_lengths_d,
            streamline_lengths_host.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
    )
    for ptr in (
        seeds_d,
        offsets_d,
        dirs_d,
        seed_of_streamline_d,
        streamline_lengths_d,
        streamline_buffer_d,
    ):
        _check(runtime.cudaFree(ptr))


def cuda_streamline_generator(simple_tracker_data, *, n_gpus=1):
    """
    Set up the CUDA probabilistic streamline generation kernel.

    Compiles the kernel and uploads the static data (PMF, stopping map,
    sphere) to each GPU.

    Parameters
    ----------
    simple_tracker_data : _SimpleTrackerData
        Output of :func:`dipy.tracking.simplet.tracker.prepare_simple_tracker_data`
    n_gpus : int, optional
        Number of GPUs to split each chunk of seeds across.
    """
    tracker_data = simple_tracker_data
    n_gpus = int(n_gpus)

    _check(driver.cuInit(0))
    n_available = _check(runtime.cudaGetDeviceCount())
    if n_gpus > n_available:
        raise RuntimeError(f"Requested {n_gpus} GPUs but only {n_available} available")
    for device_id in range(n_gpus):
        device = _check(driver.cuDeviceGet(device_id))
        try:
            ctx_params = driver.CUctxCreateParams()
            _check(driver.cuCtxCreate(ctx_params, 0, device))
        except TypeError:
            _check(driver.cuCtxCreate(0, device))

    kernel = _compile(tracker_data)

    gpus = []
    for device_id in range(n_gpus):
        _check(runtime.cudaSetDevice(device_id))
        stream = _check(
            runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
        )
        tex, arr = _allocate_texture_border(tracker_data.stop_map)
        gpus.append(
            (
                stream,
                _to_device(tracker_data.pmf),
                tex,
                arr,
                _to_device(tracker_data.sphere_vertices),
            )
        )

    def generate_streamlines(seeds, streamline_offsets, peak_dirs):
        n_seeds = len(seeds)
        step = tracker_data.max_steps * 2
        n_streamlines = int(streamline_offsets[-1])
        streamline_buffer = np.empty((n_streamlines * step, 3), dtype=KERNEL_DTYPE)
        streamline_lengths = np.zeros(n_streamlines, dtype=np.int32)

        seeds_per_gpu = _ceil_div(n_seeds, n_gpus)
        for device_id, (stream, pmf_d, tex, _, vertices_d) in enumerate(gpus):
            seed_start = device_id * seeds_per_gpu
            seed_stop = min(n_seeds, seed_start + seeds_per_gpu)
            if seed_stop <= seed_start:
                continue
            _check(runtime.cudaSetDevice(device_id))
            offsets = np.asarray(
                streamline_offsets[seed_start : seed_stop + 1]
                - streamline_offsets[seed_start],
                order="C",
            )
            sl_start = int(streamline_offsets[seed_start])
            sl_stop = int(streamline_offsets[seed_stop])
            _launch(
                kernel,
                stream,
                (pmf_d, tex, vertices_d),
                seeds[seed_start:seed_stop],
                offsets,
                peak_dirs[sl_start:sl_stop],
                streamline_lengths[sl_start:sl_stop],
                streamline_buffer[sl_start * step : sl_stop * step],
            )
        return streamline_buffer, streamline_lengths, n_streamlines

    def close():
        while gpus:
            stream, pmf_d, tex, arr, vertices_d = gpus.pop()
            device_id = len(gpus)
            _check(runtime.cudaSetDevice(device_id), hard_error=False)
            _check(runtime.cudaFree(pmf_d), hard_error=False)
            _check(runtime.cudaDestroyTextureObject(tex), hard_error=False)
            _check(runtime.cudaFreeArray(arr), hard_error=False)
            _check(runtime.cudaFree(vertices_d), hard_error=False)
            _check(runtime.cudaStreamDestroy(stream), hard_error=False)

    return generate_streamlines, close
