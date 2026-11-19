"""
WebGPU backend for :mod:`dipy.tracking.accel.tracker`.
"""

from importlib.resources import files
from time import time

import numpy as np

from dipy.tracking.accel.tracker import ceil_div, kernel_params
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package

wgpu, have_wgpu, _ = optional_package(
    "wgpu",
    trip_msg=(
        'WebGPU tracking requires wgpu. Install it with: pip install "dipy[webgpu]"'
    ),
)

KERNEL_DTYPE = np.float32
THREADS_PER_STREAMLINE = 32
STREAMLINES_PER_BLOCK = 2
# entry points in kernel.wgsl
KERNEL_NAME = "genStreamlinesProb_k"
COMPACT_KERNEL_NAME = "compactStreamlines_k"
COMPACT_WORKGROUP = 64
# WebGPU caps a dispatch at 65535 workgroups per dimension
MAX_SEEDS = 65535 * STREAMLINES_PER_BLOCK


def webgpu_available():
    if not have_wgpu:
        return False
    try:
        adapter = wgpu.gpu.request_adapter_sync()
    except RuntimeError:
        return False
    return adapter is not None and "subgroup" in adapter.features


def _data_buffer(device, arr, *, label=""):
    # a byte view of the array, rather than a copy from tobytes()
    return device.create_buffer_with_data(
        data=memoryview(np.ascontiguousarray(arr)).cast("B"),
        usage="STORAGE | COPY_SRC",
        label=label,
    )


def _upload_slabs(device, arr, *, dtype, label=""):
    """Upload ``arr`` converted to ``dtype``, one slab of its first axis at a
    time, so that the full converted host copy (as large as the PMF) is never
    allocated."""
    arr = np.asarray(arr)
    itemsize = np.dtype(dtype).itemsize
    buf = _empty_buffer(device, arr.size * itemsize, label=label)
    row_bytes = (arr.size // max(arr.shape[0], 1)) * itemsize
    rows = max(1, (64 * 1024 * 1024) // max(row_bytes, 1))
    for start in range(0, arr.shape[0], rows):
        slab = np.ascontiguousarray(arr[start : start + rows], dtype=dtype)
        device.queue.write_buffer(buf, start * row_bytes, memoryview(slab).cast("B"))
    return buf


def _empty_buffer(device, nbytes, *, label=""):
    return device.create_buffer(
        size=max(int(nbytes), 4),
        usage="STORAGE | COPY_SRC | COPY_DST",
        label=label,
    )


def _setup_device():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No WebGPU adapter found")
    if "subgroup" not in adapter.features:
        raise RuntimeError(
            "WebGPU adapter does not support subgroup operations, which the "
            "GPU tracker kernel requires. Upgrade your GPU driver or use "
            "another backend."
        )
    features = ["subgroup"]
    if "subgroup-barrier" in adapter.features:
        features.append("subgroup-barrier")
    # the spec default limits (256 MB buffers) are too small for whole-brain PMFs
    device = adapter.request_device_sync(
        required_features=features,
        required_limits={
            "max-buffer-size": adapter.limits["max-buffer-size"],
            "max-storage-buffer-binding-size": adapter.limits[
                "max-storage-buffer-binding-size"
            ],
        },
    )
    info = adapter.info
    logger.info(
        "WebGPU GPU tracker on %s (%s)",
        getattr(info, "device", "unknown"),
        getattr(info, "backend_type", "unknown"),
    )
    return device


def _compile(device, tracker_data):
    start = time()
    logger.info("Compiling WebGPU GPU tracker kernel...")
    dim_t = tracker_data.shape[3]
    n32_dim_t = ceil_div(dim_t, 32) * 32
    source = files("dipy.tracking.accel").joinpath("kernel.wgsl").read_text()
    source = (
        f"const SPHERE_SYMM: u32 = {1 if tracker_data.is_symmetric else 0}u;\n"
        f"const N32DIMT: u32 = {n32_dim_t}u;\n" + source
    )
    module = device.create_shader_module(code=source)
    pipelines = [
        device.create_compute_pipeline(
            layout="auto", compute={"module": module, "entry_point": name}
        )
        for name in (KERNEL_NAME, COMPACT_KERNEL_NAME)
    ]
    logger.info("WebGPU GPU tracker kernel compiled in %.2f s", time() - start)
    return pipelines


class _BufferPool:
    """Storage buffers reused from batch to batch, grown when too small."""

    def __init__(self, device):
        self.device = device
        self.buffers = {}

    def get(self, name, nbytes):
        nbytes = max(4, ceil_div(int(nbytes), 4) * 4)
        buf = self.buffers.get(name)
        if buf is None or buf.size < nbytes:
            if buf is not None:
                buf.destroy()
            buf = _empty_buffer(self.device, nbytes, label=name)
            self.buffers[name] = buf
        return buf

    def write(self, name, arr):
        arr = np.ascontiguousarray(arr)
        buf = self.get(name, arr.nbytes)
        if arr.nbytes:
            self.device.queue.write_buffer(buf, 0, memoryview(arr).cast("B"))
        return buf

    def destroy(self):
        for buf in self.buffers.values():
            buf.destroy()
        self.buffers.clear()


def _bind_group(device, pipeline, group, entries):
    return device.create_bind_group(
        layout=pipeline.get_bind_group_layout(group),
        entries=[
            {"binding": i, "resource": {"buffer": buf}} for i, buf in entries.items()
        ],
    )


def _workgroup_grid(n):
    """2D dispatch size for ``n`` workgroups (65535 per dimension at most)."""
    x = min(max(n, 1), 65535)
    return x, ceil_div(max(n, 1), x)


def webgpu_streamline_generator(tracker_data):
    """
    Set up the WebGPU probabilistic streamline generation kernel on the
    default WebGPU adapter.

    The streamlines are packed on the device before being read back, and
    the batch buffers are reused, so that the host only copies the points
    actually generated.

    See :func:`dipy.tracking.accel.tracker._streamline_generator` for the
    returned ``generate_streamlines`` contract.
    """
    device = _setup_device()
    step = tracker_data.max_steps * 2

    max_binding = min(
        device.limits["max-buffer-size"],
        device.limits["max-storage-buffer-binding-size"],
    )
    pmf_nbytes = tracker_data.pmf.size * np.dtype(KERNEL_DTYPE).itemsize
    if pmf_nbytes > max_binding:
        raise RuntimeError(
            f"PMF ({pmf_nbytes / 1e9:.1f} GB) exceeds the WebGPU "
            f"buffer limit ({max_binding / 1e9:.1f} GB). "
            "Use a smaller volume or fewer sphere directions."
        )

    track_pipeline, compact_pipeline = _compile(device, tracker_data)
    static_buffers = [
        _upload_slabs(device, tracker_data.pmf, dtype=KERNEL_DTYPE, label="pmf"),
        _data_buffer(
            device, tracker_data.stop_map.astype(KERNEL_DTYPE), label="stop_map"
        ),
        _data_buffer(
            device,
            tracker_data.sphere_vertices.astype(KERNEL_DTYPE),
            label="sphere_vertices",
        ),
    ]
    pool = _BufferPool(device)

    def generate_streamlines(seeds, streamline_offsets, peak_dirs, sl_base):
        n_seeds = len(seeds)
        n_streamlines = int(streamline_offsets[-1])
        streamline_nbytes = 4 * 3 * step * n_streamlines
        if streamline_nbytes > max_binding:
            max_streamlines = max_binding // (4 * 3 * step)
            raise RuntimeError(
                f"Streamline buffer ({streamline_nbytes / 1e9:.1f} GB for "
                f"{n_streamlines} streamlines) exceeds the WebGPU buffer limit "
                f"({max_binding / 1e9:.1f} GB). Reduce chunk_size so that "
                f"at most ~{max_streamlines} streamlines are generated per chunk."
            )

        params_buf = pool.write(
            "params", kernel_params(tracker_data, n_seeds=n_seeds, sl_base=sl_base)
        )
        seeds_buf = pool.write("seeds", seeds)
        offsets_buf = pool.write("offsets", streamline_offsets)
        dirs_buf = pool.write("seed_directions", peak_dirs)
        starts_buf = pool.get("streamline_starts", 4 * n_streamlines)
        lengths_buf = pool.get("streamline_lengths", 4 * n_streamlines)
        sline_buf = pool.get("streamlines", streamline_nbytes)

        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(track_pipeline)
        cpass.set_bind_group(
            0,
            _bind_group(
                device,
                track_pipeline,
                0,
                dict(enumerate([params_buf, seeds_buf, *static_buffers])),
            ),
        )
        cpass.set_bind_group(
            1,
            _bind_group(
                device,
                track_pipeline,
                1,
                dict(
                    enumerate(
                        [offsets_buf, dirs_buf, starts_buf, lengths_buf, sline_buf]
                    )
                ),
            ),
        )
        cpass.dispatch_workgroups(ceil_div(n_seeds, STREAMLINES_PER_BLOCK), 1, 1)
        cpass.end()
        device.queue.submit([encoder.finish()])

        # small reads, which also wait for the tracking kernel
        lengths = np.frombuffer(
            device.queue.read_buffer(lengths_buf, size=4 * n_streamlines),
            dtype=np.int32,
        ).copy()
        rows = np.zeros(n_streamlines + 1, dtype=np.uint32)
        np.cumsum(np.abs(lengths), out=rows[1:])
        n_points = int(rows[-1])

        out_offsets_buf = pool.write("out_offsets", rows)
        packed_buf = pool.get("packed", 4 * 3 * n_points)
        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(compact_pipeline)
        cpass.set_bind_group(
            0, _bind_group(device, compact_pipeline, 0, {0: params_buf})
        )
        cpass.set_bind_group(
            1,
            _bind_group(
                device,
                compact_pipeline,
                1,
                {2: starts_buf, 3: lengths_buf, 4: sline_buf},
            ),
        )
        cpass.set_bind_group(
            2,
            _bind_group(
                device, compact_pipeline, 2, {0: out_offsets_buf, 1: packed_buf}
            ),
        )
        cpass.dispatch_workgroups(
            *_workgroup_grid(ceil_div(n_streamlines, COMPACT_WORKGROUP)), 1
        )
        cpass.end()
        device.queue.submit([encoder.finish()])

        packed = np.frombuffer(
            device.queue.read_buffer(packed_buf, size=max(4, 4 * 3 * n_points)),
            dtype=KERNEL_DTYPE,
        )[: 3 * n_points].reshape(n_points, 3)
        return packed, rows[:-1].astype(np.intp), lengths

    def close():
        pool.destroy()
        for buf in static_buffers:
            buf.destroy()
        static_buffers.clear()

    return generate_streamlines, close
