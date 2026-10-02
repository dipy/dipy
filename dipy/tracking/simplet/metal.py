"""
Metal (Apple Silicon) backend for :mod:`dipy.tracking.simplet.tracker`.
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

KERNEL_DTYPE = np.float32
THREADS_PER_STREAMLINE = 32
STREAMLINES_PER_BLOCK = 2
# symbol name in kernel.metal
KERNEL_NAME = "genStreamlinesProb_k"


def metal_available():
    return have_metal and Metal.MTLCreateSystemDefaultDevice() is not None


def _ceil_div(a, b):
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


def _compile(device, tracker_data):
    start = time()
    logger.info("Compiling Metal simple tracker kernel...")
    dim_t = tracker_data.shape[3]
    n32_dim_t = _ceil_div(dim_t, 32) * 32
    source = files("dipy.tracking.simplet").joinpath("kernel.metal").read_text()
    source = (
        f"#define SPHERE_SYMM {1 if tracker_data.is_symmetric else 0}\n"
        f"#define N32DIMT {n32_dim_t}\n" + source
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


def _params_bytes(tracker_data, n_seeds):
    # Must match ProbTrackingParams in kernel.metal: 4 floats, 8 ints.
    seed = tracker_data.random_seed
    return struct.pack(
        "4f8i",
        tracker_data.max_angle,
        tracker_data.stop_threshold,
        tracker_data.step_size,
        tracker_data.pmf_threshold,
        seed & 0xFFFFFFFF,
        (seed >> 32) & 0xFFFFFFFF,
        int(n_seeds),
        *tracker_data.shape,
        tracker_data.max_steps,
    )


def metal_streamline_generator(simple_tracker_data):
    """
    Set up the Metal probabilistic streamline generation kernel on the
    default Metal device.
    """
    tracker_data = simple_tracker_data

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal GPU device found")
    command_queue = device.newCommandQueue()
    logger.info("Metal simple tracker on %s", device.name())

    if tracker_data.pmf.nbytes > device.maxBufferLength():
        raise RuntimeError(
            f"PMF ({tracker_data.pmf.nbytes / 1e9:.1f} GB) exceeds the Metal "
            f"buffer limit ({device.maxBufferLength() / 1e9:.1f} GB). "
            "Use a smaller volume or fewer sphere directions."
        )

    pipeline = _compile(device, tracker_data)
    static_buffers = [
        _shared_buffer(device, tracker_data.pmf),
        _shared_buffer(device, tracker_data.stop_map),
        _shared_buffer(device, tracker_data.sphere_vertices),
    ]

    def generate_streamlines(seeds, streamline_offsets, peak_dirs):
        n_seeds = len(seeds)
        step = tracker_data.max_steps * 2
        n_streamlines = int(streamline_offsets[-1])
        if n_seeds == 0 or n_streamlines == 0:
            return np.empty((0, 3), dtype=KERNEL_DTYPE), np.zeros(0, dtype=np.int32), 0

        seeds_buf = _shared_buffer(device, seeds)
        offsets_buf = _shared_buffer(device, streamline_offsets)
        dirs_buf = _shared_buffer(device, peak_dirs)
        seed_of_streamline_buf = _empty_buffer(device, 4 * n_streamlines)
        streamline_lengths_buf = _empty_buffer(device, 4 * n_streamlines)
        streamline_buf = _empty_buffer(device, 4 * 3 * step * n_streamlines)
        _buffer_as_array(streamline_lengths_buf, np.int32, (n_streamlines,))[:] = 0

        params = _params_bytes(tracker_data, n_seeds)
        cmd_buf = command_queue.commandBuffer()
        enc = cmd_buf.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        enc.setBytes_length_atIndex_(params, len(params), 0)
        bufs = [
            seeds_buf,
            *static_buffers,
            offsets_buf,
            dirs_buf,
            seed_of_streamline_buf,
            streamline_lengths_buf,
            streamline_buf,
        ]
        for idx, buf in enumerate(bufs, start=1):
            enc.setBuffer_offset_atIndex_(buf, 0, idx)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSize(_ceil_div(n_seeds, STREAMLINES_PER_BLOCK), 1, 1),
            Metal.MTLSize(THREADS_PER_STREAMLINE, STREAMLINES_PER_BLOCK, 1),
        )
        enc.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        if cmd_buf.status() == Metal.MTLCommandBufferStatusError:
            raise RuntimeError(f"Metal command buffer error: {cmd_buf.error()}")

        streamline_buffer = _buffer_as_array(
            streamline_buf, KERNEL_DTYPE, (n_streamlines * step, 3)
        ).copy()
        streamline_lengths = _buffer_as_array(
            streamline_lengths_buf, np.int32, (n_streamlines,)
        ).copy()
        return streamline_buffer, streamline_lengths, n_streamlines

    def close():
        # Metal buffers are reference counted; dropping references frees them.
        static_buffers.clear()

    return generate_streamlines, close
