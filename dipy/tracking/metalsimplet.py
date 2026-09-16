"""
Metal (Apple Silicon) backend for :mod:`dipy.tracking.simpletracker`.
"""

from importlib.resources import files
import logging
import struct
from time import time

import numpy as np

from dipy.utils.optpkg import optional_package

logger = logging.getLogger("dipy")

Metal, have_metal, _ = optional_package(
    "Metal",
    trip_msg=(
        "Metal tracking requires pyobjc-framework-Metal and Apple Silicon. "
        'Install it with: pip install "dipy[metal]"'
    ),
)

REAL_DTYPE = np.float32
THR_X_SL = 32
BLOCK_Y = 2
KERNEL_NAME = "genStreamlinesProb_k"


def metal_available():
    return have_metal and Metal.MTLCreateSystemDefaultDevice() is not None


def _div_up(a, b):
    return (a + b - 1) // b


def _shared_buffer(device, arr):
    arr = np.asarray(arr, order="C")
    return device.newBufferWithBytes_length_options_(
        arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared
    )


def _empty_buffer(device, nbytes):
    return device.newBufferWithLength_options_(
        max(int(nbytes), 1), Metal.MTLResourceStorageModeShared
    )


def _buffer_as_array(buf, dtype, shape):
    count = int(np.prod(shape))
    memview = buf.contents().as_buffer(buf.length())
    return np.frombuffer(memview, dtype=dtype, count=count).reshape(shape)


def _compile(device, std):
    start = time()
    logger.info("Compiling Metal simple tracker kernel...")
    n32dimt = _div_up(std.dimt, 32) * 32
    source = files("dipy.tracking").joinpath("simplet.metal").read_text()
    source = (
        f"#define SPHERE_SYMM {1 if std.sphere_symm else 0}\n"
        f"#define N32DIMT {n32dimt}\n" + source
    )
    options = Metal.MTLCompileOptions.new()
    options.setFastMathEnabled_(True)
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    if error is not None:
        raise RuntimeError(f"Metal shader compilation failed: {error}")
    fn = library.newFunctionWithName_(KERNEL_NAME)
    if fn is None:
        raise RuntimeError(f"Metal kernel {KERNEL_NAME!r} not found")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    if error is not None:
        raise RuntimeError(f"Failed to create pipeline for {KERNEL_NAME!r}: {error}")
    logger.info("Metal simple tracker kernel compiled in %.2f s", time() - start)
    return pipeline


def _params_bytes(std, nseed):
    # Must match ProbTrackingParams in simplet.metal: 4 floats, 8 ints.
    seed = int(std.random_seed)
    return struct.pack(
        "4f8i",
        float(std.max_angle),
        float(std.stop_threshold),
        float(std.step_size),
        float(std.pmf_threshold),
        seed & 0xFFFFFFFF,
        (seed >> 32) & 0xFFFFFFFF,
        int(nseed),
        int(std.dimx),
        int(std.dimy),
        int(std.dimz),
        int(std.dimt),
        int(std.max_sline_len),
    )


def metal_gen_streamlines_prob(simple_tracker_data):
    """
    Set up the Metal probabilistic streamline generation kernel on the
    default Metal device.
    """
    std = simple_tracker_data

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal GPU device found")
    command_queue = device.newCommandQueue()
    logger.info("Metal simple tracker on %s", device.name())

    if std.dataf.nbytes > device.maxBufferLength():
        raise RuntimeError(
            f"PMF ({std.dataf.nbytes / 1e9:.1f} GB) exceeds the Metal buffer "
            f"limit ({device.maxBufferLength() / 1e9:.1f} GB). "
            "Use a smaller volume or fewer sphere directions."
        )

    pipeline = _compile(device, std)
    static = [
        _shared_buffer(device, std.dataf),
        _shared_buffer(device, std.metric_map),
        _shared_buffer(device, std.sphere_vertices),
    ]

    def gen_streamlines(seeds, sline_offsets, peak_dirs):
        nseed = len(seeds)
        step = std.max_sline_len * 2
        n_slines = int(sline_offsets[-1])
        if nseed == 0 or n_slines == 0:
            return np.empty((0, 3), dtype=REAL_DTYPE), np.zeros(0, dtype=np.int32), 0

        seeds_buf = _shared_buffer(device, seeds)
        offsets_buf = _shared_buffer(device, sline_offsets)
        dirs_buf = _shared_buffer(device, peak_dirs)
        sline_seed_buf = _empty_buffer(device, 4 * n_slines)
        sline_len_buf = _empty_buffer(device, 4 * n_slines)
        sline_buf = _empty_buffer(device, 4 * 3 * step * n_slines)
        _buffer_as_array(sline_len_buf, np.int32, (n_slines,))[:] = 0

        params = _params_bytes(std, nseed)
        cmd_buf = command_queue.commandBuffer()
        enc = cmd_buf.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        enc.setBytes_length_atIndex_(params, len(params), 0)
        bufs = [
            seeds_buf,
            *static,
            offsets_buf,
            dirs_buf,
            sline_seed_buf,
            sline_len_buf,
            sline_buf,
        ]
        for idx, buf in enumerate(bufs, start=1):
            enc.setBuffer_offset_atIndex_(buf, 0, idx)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSize(_div_up(nseed, BLOCK_Y), 1, 1),
            Metal.MTLSize(THR_X_SL, BLOCK_Y, 1),
        )
        enc.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        if cmd_buf.status() == Metal.MTLCommandBufferStatusError:
            raise RuntimeError(f"Metal command buffer error: {cmd_buf.error()}")

        sline = _buffer_as_array(sline_buf, REAL_DTYPE, (n_slines * step, 3)).copy()
        sline_len = _buffer_as_array(sline_len_buf, np.int32, (n_slines,)).copy()
        return sline, sline_len, n_slines

    def close():
        # Metal buffers are reference counted; dropping references frees them.
        static.clear()

    return gen_streamlines, close
