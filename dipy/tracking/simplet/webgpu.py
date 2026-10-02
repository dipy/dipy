"""
WebGPU backend for :mod:`dipy.tracking.simplet.tracker`.
"""

from importlib.resources import files
import logging
from time import time

import numpy as np

from dipy.utils.optpkg import optional_package

logger = logging.getLogger("dipy")

wgpu, have_wgpu, _ = optional_package(
    "wgpu",
    trip_msg=(
        'WebGPU tracking requires wgpu. Install it with: pip install "dipy[webgpu]"'
    ),
)

KERNEL_DTYPE = np.float32
THREADS_PER_STREAMLINE = 32
STREAMLINES_PER_BLOCK = 2
# entry point in kernel.wgsl
KERNEL_NAME = "genStreamlinesProb_k"


def webgpu_available():
    if not have_wgpu:
        return False
    try:
        adapter = wgpu.gpu.request_adapter_sync()
    except RuntimeError:
        return False
    return adapter is not None and "subgroup" in adapter.features


def _ceil_div(a, b):
    return (a + b - 1) // b


def _data_buffer(device, arr, *, label=""):
    return device.create_buffer_with_data(
        data=np.asarray(arr, order="C").tobytes(),
        usage="STORAGE | COPY_SRC",
        label=label,
    )


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
            "simple tracker kernel requires. Upgrade your GPU driver or use "
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
        "WebGPU simple tracker on %s (%s)",
        getattr(info, "device", "unknown"),
        getattr(info, "backend_type", "unknown"),
    )
    return device


def _compile(device, tracker_data):
    start = time()
    logger.info("Compiling WebGPU simple tracker kernel...")
    dim_t = tracker_data.shape[3]
    n32_dim_t = _ceil_div(dim_t, 32) * 32
    source = files("dipy.tracking.simplet").joinpath("kernel.wgsl").read_text()
    source = (
        f"const SPHERE_SYMM: u32 = {1 if tracker_data.is_symmetric else 0}u;\n"
        f"const N32DIMT: u32 = {n32_dim_t}u;\n" + source
    )
    module = device.create_shader_module(code=source)
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": KERNEL_NAME}
    )
    logger.info("WebGPU simple tracker kernel compiled in %.2f s", time() - start)
    return pipeline


def _params(tracker_data, n_seeds):
    # Must match ProbTrackingParams in kernel.wgsl: 4 f32, 8 i32.
    seed = tracker_data.random_seed
    params = np.zeros(12, dtype=np.uint32)
    params[:4] = np.array(
        [
            tracker_data.max_angle,
            tracker_data.stop_threshold,
            tracker_data.step_size,
            tracker_data.pmf_threshold,
        ],
        dtype=np.float32,
    ).view(np.uint32)
    params[4:] = [
        seed & 0xFFFFFFFF,
        (seed >> 32) & 0xFFFFFFFF,
        n_seeds,
        *tracker_data.shape,
        tracker_data.max_steps,
    ]
    return params


def webgpu_streamline_generator(simple_tracker_data):
    """
    Set up the WebGPU probabilistic streamline generation kernel on the
    default WebGPU adapter.

    See :func:`dipy.tracking.simplet.cuda.cuda_streamline_generator` for the
    returned ``(generate_streamlines, close)`` contract.
    """
    tracker_data = simple_tracker_data
    device = _setup_device()

    max_binding = min(
        device.limits["max-buffer-size"],
        device.limits["max-storage-buffer-binding-size"],
    )
    if tracker_data.pmf.nbytes > max_binding:
        raise RuntimeError(
            f"PMF ({tracker_data.pmf.nbytes / 1e9:.1f} GB) exceeds the WebGPU "
            f"buffer limit ({max_binding / 1e9:.1f} GB). "
            "Use a smaller volume or fewer sphere directions."
        )

    pipeline = _compile(device, tracker_data)
    static_buffers = [
        _data_buffer(device, tracker_data.pmf, label="pmf"),
        _data_buffer(device, tracker_data.stop_map, label="stop_map"),
        _data_buffer(device, tracker_data.sphere_vertices, label="sphere_vertices"),
    ]

    def generate_streamlines(seeds, streamline_offsets, peak_dirs):
        n_seeds = len(seeds)
        step = tracker_data.max_steps * 2
        n_streamlines = int(streamline_offsets[-1])
        if n_seeds == 0 or n_streamlines == 0:
            return np.empty((0, 3), dtype=KERNEL_DTYPE), np.zeros(0, dtype=np.int32), 0

        streamline_nbytes = 4 * 3 * step * n_streamlines
        if streamline_nbytes > max_binding:
            max_streamlines = max_binding // (4 * 3 * step)
            raise RuntimeError(
                f"Streamline buffer ({streamline_nbytes / 1e9:.1f} GB for "
                f"{n_streamlines} streamlines) exceeds the WebGPU buffer limit "
                f"({max_binding / 1e9:.1f} GB). Reduce chunk_size so that "
                f"at most ~{max_streamlines} streamlines are generated per chunk."
            )

        params_buf = _data_buffer(
            device, _params(tracker_data, n_seeds), label="params"
        )
        seeds_buf = _data_buffer(device, seeds, label="seeds")
        offsets_buf = _data_buffer(device, streamline_offsets, label="offsets")
        dirs_buf = _data_buffer(device, peak_dirs, label="seed_directions")
        seed_of_streamline_buf = _empty_buffer(
            device, 4 * n_streamlines, label="seed_of_streamline"
        )
        streamline_lengths_buf = _empty_buffer(
            device, 4 * n_streamlines, label="streamline_lengths"
        )
        streamline_buf = _empty_buffer(device, streamline_nbytes, label="streamlines")
        device.queue.write_buffer(
            streamline_lengths_buf, 0, np.zeros(n_streamlines, np.int32)
        )

        def bind_group(group, bufs):
            return device.create_bind_group(
                layout=pipeline.get_bind_group_layout(group),
                entries=[
                    {"binding": i, "resource": {"buffer": b}}
                    for i, b in enumerate(bufs)
                ],
            )

        bg0 = bind_group(0, [params_buf, seeds_buf, *static_buffers])
        bg1 = bind_group(
            1,
            [
                offsets_buf,
                dirs_buf,
                seed_of_streamline_buf,
                streamline_lengths_buf,
                streamline_buf,
            ],
        )

        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(pipeline)
        cpass.set_bind_group(0, bg0)
        cpass.set_bind_group(1, bg1)
        cpass.dispatch_workgroups(_ceil_div(n_seeds, STREAMLINES_PER_BLOCK), 1, 1)
        cpass.end()
        device.queue.submit([encoder.finish()])

        streamline_buffer = np.frombuffer(
            device.queue.read_buffer(streamline_buf), dtype=KERNEL_DTYPE
        )
        streamline_buffer = streamline_buffer[: n_streamlines * step * 3].reshape(
            n_streamlines * step, 3
        )
        streamline_lengths = np.frombuffer(
            device.queue.read_buffer(streamline_lengths_buf), dtype=np.int32
        )[:n_streamlines]
        return streamline_buffer, streamline_lengths, n_streamlines

    def close():
        static_buffers.clear()

    return generate_streamlines, close
