"""
CUDA backend for :mod:`dipy.tracking.simpletracker`.
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

REAL_DTYPE = np.float32
REAL_SIZE = 4
INT_SIZE = 4
THR_X_SL = 32  # threads per streamline
THR_X_BL = 64  # threads per block
BLOCK_Y = THR_X_BL // THR_X_SL
KERNEL_NAME = f"genStreamlinesProb_k<{THR_X_SL},{BLOCK_Y}>"


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


def _div_up(a, b):
    return (a + b - 1) // b


def _to_device(host):
    host = np.ascontiguousarray(host)
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


def _compile(std):
    start = time()
    logger.info("Compiling CUDA simple tracker kernel...")
    macros = {
        "DIMX": int(std.dimx),
        "DIMY": int(std.dimy),
        "DIMZ": int(std.dimz),
        "DIMT": int(std.dimt),
        "STEP_SIZE": float(std.step_size),
        "MAX_ANGLE": float(std.max_angle),
        "TC_THRESHOLD": float(std.stop_threshold),
        "PMF_THRESHOLD_P": float(std.pmf_threshold),
        "MAX_SLINE_LEN": int(std.max_sline_len),
        "RNG_SEED": int(std.random_seed),
        "SPHERE_SYMM": 1 if std.sphere_symm else 0,
        "THR_X_SL": THR_X_SL,
        "THR_X_BL": THR_X_BL,
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
    source = files("dipy.tracking").joinpath("simplet.cu").read_text()
    prog = core.Program(source, code_type="c++", options=options)
    module = prog.compile("cubin", name_expressions=(KERNEL_NAME,))
    logger.info("CUDA simple tracker kernel compiled in %.2f s", time() - start)
    return module.get_kernel(KERNEL_NAME)


def _launch(
    kernel, stream, static, seeds, sline_offsets, peak_dirs, sline_len_host, sline_host
):
    dataf_d, metric_map_tex, sphere_vertices_d = static
    nseed = len(seeds)
    n_slines = int(sline_offsets[-1])
    if nseed == 0 or n_slines == 0:
        return

    seeds_d = _to_device(seeds)
    offsets_d = _to_device(sline_offsets)
    dirs_d = _to_device(peak_dirs)
    sline_seed_d = _check(runtime.cudaMalloc(INT_SIZE * n_slines))
    sline_len_d = _check(runtime.cudaMalloc(INT_SIZE * n_slines))
    sline_d = _check(runtime.cudaMalloc(sline_host.nbytes))

    config = core.LaunchConfig(
        block=(THR_X_SL, BLOCK_Y, 1), grid=(_div_up(nseed, BLOCK_Y), 1, 1), shmem_size=0
    )
    core.launch(
        stream,
        config,
        kernel,
        nseed,
        seeds_d,
        dataf_d,
        metric_map_tex.getPtr(),
        sphere_vertices_d,
        offsets_d,
        dirs_d,
        sline_seed_d,
        sline_len_d,
        sline_d,
    )
    _check(runtime.cudaStreamSynchronize(stream))

    _check(
        runtime.cudaMemcpy(
            sline_host.ctypes.data,
            sline_d,
            sline_host.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
    )
    _check(
        runtime.cudaMemcpy(
            sline_len_host.ctypes.data,
            sline_len_d,
            sline_len_host.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
    )
    for ptr in (seeds_d, offsets_d, dirs_d, sline_seed_d, sline_len_d, sline_d):
        _check(runtime.cudaFree(ptr))


def cuda_gen_streamlines_prob(simple_tracker_data, *, ngpus=1):
    """
    Set up the CUDA probabilistic streamline generation kernel.

    Compiles the kernel and uploads the static data (PMF, stopping map,
    sphere) to each GPU.

    Parameters
    ----------
    simple_tracker_data : _SimpleTrackerData
        Output of :func:`dipy.tracking.simpletracker.prepare_simple_tracker_data`
    ngpus : int, optional
        Number of GPUs to split each chunk of seeds across.
    """
    std = simple_tracker_data
    ngpus = int(ngpus)

    _check(driver.cuInit(0))
    avail = _check(runtime.cudaGetDeviceCount())
    if ngpus > avail:
        raise RuntimeError(f"Requested {ngpus} GPUs but only {avail} available")
    for ii in range(ngpus):
        device = _check(driver.cuDeviceGet(ii))
        try:
            ctx_params = driver.CUctxCreateParams()
            _check(driver.cuCtxCreate(ctx_params, 0, device))
        except TypeError:
            _check(driver.cuCtxCreate(0, device))

    kernel = _compile(std)

    gpus = []
    for ii in range(ngpus):
        _check(runtime.cudaSetDevice(ii))
        stream = _check(
            runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
        )
        tex, arr = _allocate_texture_border(std.metric_map)
        gpus.append(
            (stream, _to_device(std.dataf), tex, arr, _to_device(std.sphere_vertices))
        )

    def gen_streamlines(seeds, sline_offsets, peak_dirs):
        nseed = len(seeds)
        step = std.max_sline_len * 2
        n_slines = int(sline_offsets[-1])
        sline = np.empty((n_slines * step, 3), dtype=REAL_DTYPE)
        sline_len = np.zeros(n_slines, dtype=np.int32)

        per_gpu = _div_up(nseed, ngpus)
        for ii, (stream, dataf_d, tex, _, verts_d) in enumerate(gpus):
            a = ii * per_gpu
            b = min(nseed, a + per_gpu)
            if b <= a:
                continue
            _check(runtime.cudaSetDevice(ii))
            offs = np.ascontiguousarray(sline_offsets[a : b + 1] - sline_offsets[a])
            sl_a = int(sline_offsets[a])
            sl_b = int(sline_offsets[b])
            _launch(
                kernel,
                stream,
                (dataf_d, tex, verts_d),
                seeds[a:b],
                offs,
                peak_dirs[a * std.dimt : b * std.dimt],
                sline_len[sl_a:sl_b],
                sline[sl_a * step : sl_b * step],
            )
        return sline, sline_len, n_slines

    def close():
        while gpus:
            stream, dataf_d, tex, arr, verts_d = gpus.pop()
            ii = len(gpus)
            _check(runtime.cudaSetDevice(ii), hard_error=False)
            _check(runtime.cudaFree(dataf_d), hard_error=False)
            _check(runtime.cudaDestroyTextureObject(tex), hard_error=False)
            _check(runtime.cudaFreeArray(arr), hard_error=False)
            _check(runtime.cudaFree(verts_d), hard_error=False)
            _check(runtime.cudaStreamDestroy(stream), hard_error=False)

    return gen_streamlines, close
