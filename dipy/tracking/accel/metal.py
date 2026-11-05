"""
Metal (Apple Silicon) backend for :mod:`dipy.tracking.accel.tracker`.
"""

from contextlib import nullcontext
from importlib.resources import files
import logging
import struct
from time import time

import numpy as np

from dipy.utils.optpkg import optional_package

logger = logging.getLogger("dipy")

_TRIP_MSG = (
    "Metal tracking requires pyobjc-framework-Metal and Apple Silicon. "
    'Install it with: pip install "dipy[metal]"'
)
Metal, have_metal, _ = optional_package("Metal", trip_msg=_TRIP_MSG)
objc, have_objc, _ = optional_package("objc", trip_msg=_TRIP_MSG)

KERNEL_DTYPE = np.float32
THREADS_PER_STREAMLINE = 32
STREAMLINES_PER_BLOCK = 2
# symbol name in kernel.metal
KERNEL_NAME = "genStreamlinesProb_k"


def metal_available():
    return have_metal and Metal.MTLCreateSystemDefaultDevice() is not None


def _ceil_div(a, b):
    return (a + b - 1) // b


def _autorelease_pool():
    # Objects returned autoreleased by Metal (command buffers, encoders, ...)
    # are only freed when an autorelease pool drains, and a command buffer
    # keeps every buffer it used alive. Without a pool per chunk, all the
    # chunks' streamline buffers stay in memory until the process exits.
    return objc.autorelease_pool() if have_objc else nullcontext()


def _new_buffer(device, nbytes):
    return device.newBufferWithLength_options_(
        max(int(nbytes), 1), Metal.MTLResourceStorageModeShared
    )


def _buffer_as_array(buf, dtype, shape):
    count = int(np.prod(shape))
    memview = buf.contents().as_buffer(buf.length())
    return np.frombuffer(memview, dtype=dtype, count=count).reshape(shape)


def _upload(device, arr, *, dtype=None):
    """Copy ``arr`` (converted to ``dtype``) into a new shared buffer.

    Writing straight into the buffer avoids the temporary host copies of
    ``arr.astype(dtype).tobytes()``, which matter for the PMF.
    """
    arr = np.asarray(arr)
    dtype = arr.dtype if dtype is None else np.dtype(dtype)
    buf = _new_buffer(device, arr.size * dtype.itemsize)
    if arr.size:
        _buffer_as_array(buf, dtype, arr.shape)[...] = arr
    return buf


class _GrowableBuffer:
    """A shared Metal buffer reused across chunks, grown when too small."""

    def __init__(self, device):
        self.device = device
        self.buffer = None
        self.nbytes = 0

    def get(self, nbytes):
        nbytes = max(int(nbytes), 1)
        if nbytes > self.nbytes:
            self.buffer = None  # release the old buffer first
            self.buffer = _new_buffer(self.device, nbytes)
            self.nbytes = nbytes
        return self.buffer


def _compile(device, tracker_data):
    start = time()
    logger.info("Compiling Metal GPU tracker kernel...")
    dim_t = tracker_data.shape[3]
    n32_dim_t = _ceil_div(dim_t, 32) * 32
    source = files("dipy.tracking.accel").joinpath("kernel.metal").read_text()
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
    logger.info("Metal GPU tracker kernel compiled in %.2f s", time() - start)
    return pipeline


def _params_bytes(tracker_data, n_seeds, sl_base):
    # Must match ProbTrackingParams in kernel.metal: 4 floats, 4 uints, 6 ints.
    seed = tracker_data.random_seed
    return struct.pack(
        "4f4I6i",
        tracker_data.max_angle,
        tracker_data.stop_threshold,
        tracker_data.step_size,
        tracker_data.pmf_threshold,
        seed & 0xFFFFFFFF,
        (seed >> 32) & 0xFFFFFFFF,
        sl_base & 0xFFFFFFFF,
        (sl_base >> 32) & 0xFFFFFFFF,
        int(n_seeds),
        *tracker_data.shape,
        tracker_data.max_steps,
    )


def metal_streamline_generator(tracker_data):
    """
    Set up the Metal probabilistic streamline generation kernel on the
    default Metal device.

    See :func:`dipy.tracking.accel.tracker._streamline_generator` for the
    returned ``generate_streamlines`` contract.
    """
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal GPU device found")
    command_queue = device.newCommandQueue()
    logger.info("Metal GPU tracker on %s", device.name())

    pmf_nbytes = tracker_data.pmf.size * np.dtype(KERNEL_DTYPE).itemsize
    if pmf_nbytes > device.maxBufferLength():
        raise RuntimeError(
            f"PMF ({pmf_nbytes / 1e9:.1f} GB) exceeds the Metal "
            f"buffer limit ({device.maxBufferLength() / 1e9:.1f} GB). "
            "Use a smaller volume or fewer sphere directions."
        )

    step = tracker_data.max_steps * 2
    with _autorelease_pool():
        pipeline = _compile(device, tracker_data)
        static_buffers = [
            _upload(device, tracker_data.pmf, KERNEL_DTYPE),
            _upload(device, tracker_data.stop_map, KERNEL_DTYPE),
            _upload(device, tracker_data.sphere_vertices, KERNEL_DTYPE),
        ]
    # outputs are reused from chunk to chunk
    starts_out = _GrowableBuffer(device)
    lengths_out = _GrowableBuffer(device)
    streamlines_out = _GrowableBuffer(device)

    def generate_streamlines(seeds, streamline_offsets, peak_dirs, sl_base):
        n_seeds = len(seeds)
        n_streamlines = int(streamline_offsets[-1])
        if n_seeds == 0 or n_streamlines == 0:
            return (
                np.empty((0, 3), dtype=KERNEL_DTYPE),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
            )

        with _autorelease_pool():
            seeds_buf = _upload(device, seeds)
            offsets_buf = _upload(device, streamline_offsets)
            dirs_buf = _upload(device, peak_dirs)
            starts_buf = starts_out.get(4 * n_streamlines)
            lengths_buf = lengths_out.get(4 * n_streamlines)
            streamline_buf = streamlines_out.get(4 * 3 * step * n_streamlines)

            params = _params_bytes(tracker_data, n_seeds, sl_base)
            cmd_buf = command_queue.commandBuffer()
            enc = cmd_buf.computeCommandEncoder()
            enc.setComputePipelineState_(pipeline)
            enc.setBytes_length_atIndex_(params, len(params), 0)
            bufs = [
                seeds_buf,
                *static_buffers,
                offsets_buf,
                dirs_buf,
                starts_buf,
                lengths_buf,
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
            del enc, cmd_buf, bufs, seeds_buf, offsets_buf, dirs_buf

        # Views on the shared output buffers (no copy): they stay valid until
        # the next call, which is all the caller needs.
        return (
            _buffer_as_array(streamline_buf, KERNEL_DTYPE, (n_streamlines * step, 3)),
            _buffer_as_array(starts_buf, np.int32, (n_streamlines,)),
            _buffer_as_array(lengths_buf, np.int32, (n_streamlines,)),
        )

    def close():
        # Metal buffers are reference counted; dropping references frees them.
        static_buffers.clear()
        for out in (starts_out, lengths_out, streamlines_out):
            out.buffer = None
            out.nbytes = 0

    return generate_streamlines, close
