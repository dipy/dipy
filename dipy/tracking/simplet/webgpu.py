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

REAL_DTYPE = np.float32
THR_X_SL = 32
BLOCK_Y = 2


def webgpu_available():
    if not have_wgpu:
        return False
    try:
        adapter = wgpu.gpu.request_adapter_sync()
    except RuntimeError:
        return False
    return adapter is not None and "subgroup" in adapter.features


def _div_up(a, b):
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
    # Ask for the adapter's real limits; the spec defaults (256 MB buffers)
    # are too small for whole-brain PMFs.
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


def _compile(device, std):
    start = time()
    logger.info("Compiling WebGPU simple tracker kernel...")
    n32dimt = _div_up(std.dimt, 32) * 32
    source = files("dipy.tracking.simplet").joinpath("kernel.wgsl").read_text()
    source = (
        f"const SPHERE_SYMM: u32 = {1 if std.sphere_symm else 0}u;\n"
        f"const N32DIMT: u32 = {n32dimt}u;\n" + source
    )
    module = device.create_shader_module(code=source)
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "genStreamlinesProb_k"}
    )
    logger.info("WebGPU simple tracker kernel compiled in %.2f s", time() - start)
    return pipeline


def _params(std, nseed):
    # Must match ProbTrackingParams in kernel.wgsl: 4 f32, 8 i32.
    seed = int(std.random_seed)
    params = np.zeros(12, dtype=np.uint32)
    params[:4] = np.array(
        [std.max_angle, std.stop_threshold, std.step_size, std.pmf_threshold],
        dtype=np.float32,
    ).view(np.uint32)
    params[4:] = [
        seed & 0xFFFFFFFF,
        (seed >> 32) & 0xFFFFFFFF,
        nseed,
        std.dimx,
        std.dimy,
        std.dimz,
        std.dimt,
        std.max_sline_len,
    ]
    return params


def webgpu_gen_streamlines_prob(simple_tracker_data):
    """
    Set up the WebGPU probabilistic streamline generation kernel on the
    default WebGPU adapter.

    See :func:`dipy.tracking.simplet.cuda.cuda_gen_streamlines_prob` for the
    returned ``(gen_streamlines, close)`` contract.
    """
    std = simple_tracker_data
    device = _setup_device()

    max_binding = min(
        device.limits["max-buffer-size"],
        device.limits["max-storage-buffer-binding-size"],
    )
    if std.dataf.nbytes > max_binding:
        raise RuntimeError(
            f"PMF ({std.dataf.nbytes / 1e9:.1f} GB) exceeds the WebGPU buffer "
            f"limit ({max_binding / 1e9:.1f} GB). "
            "Use a smaller volume or fewer sphere directions."
        )

    pipeline = _compile(device, std)
    static = [
        _data_buffer(device, std.dataf, "dataf"),
        _data_buffer(device, std.metric_map, "metric_map"),
        _data_buffer(device, std.sphere_vertices, "sphere_vertices"),
    ]

    def gen_streamlines(seeds, sline_offsets, peak_dirs):
        nseed = len(seeds)
        step = std.max_sline_len * 2
        n_slines = int(sline_offsets[-1])
        if nseed == 0 or n_slines == 0:
            return np.empty((0, 3), dtype=REAL_DTYPE), np.zeros(0, dtype=np.int32), 0

        sline_nbytes = 4 * 3 * step * n_slines
        if sline_nbytes > max_binding:
            max_slines = max_binding // (4 * 3 * step)
            raise RuntimeError(
                f"Streamline buffer ({sline_nbytes / 1e9:.1f} GB for {n_slines} "
                f"streamlines) exceeds the WebGPU buffer limit "
                f"({max_binding / 1e9:.1f} GB). Reduce chunk_size so that "
                f"at most ~{max_slines} streamlines are generated per chunk."
            )

        params_buf = _data_buffer(device, _params(std, nseed), "params")
        seeds_buf = _data_buffer(device, seeds, "seeds")
        offsets_buf = _data_buffer(device, sline_offsets, "slineOutOff")
        dirs_buf = _data_buffer(device, peak_dirs, "shDir0")
        sline_seed_buf = _empty_buffer(device, 4 * n_slines, "slineSeed")
        sline_len_buf = _empty_buffer(device, 4 * n_slines, "slineLen")
        sline_buf = _empty_buffer(device, sline_nbytes, "sline")
        device.queue.write_buffer(sline_len_buf, 0, np.zeros(n_slines, np.int32))

        def bind_group(group, bufs):
            return device.create_bind_group(
                layout=pipeline.get_bind_group_layout(group),
                entries=[
                    {"binding": i, "resource": {"buffer": b}}
                    for i, b in enumerate(bufs)
                ],
            )

        bg0 = bind_group(0, [params_buf, seeds_buf, *static])
        bg1 = bind_group(
            1, [offsets_buf, dirs_buf, sline_seed_buf, sline_len_buf, sline_buf]
        )

        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(pipeline)
        cpass.set_bind_group(0, bg0)
        cpass.set_bind_group(1, bg1)
        cpass.dispatch_workgroups(_div_up(nseed, BLOCK_Y), 1, 1)
        cpass.end()
        device.queue.submit([encoder.finish()])

        sline = np.frombuffer(device.queue.read_buffer(sline_buf), dtype=REAL_DTYPE)
        sline = sline[: n_slines * step * 3].reshape(n_slines * step, 3)
        sline_len = np.frombuffer(
            device.queue.read_buffer(sline_len_buf), dtype=np.int32
        )[:n_slines]
        return sline, sline_len, n_slines

    def close():
        static.clear()

    return gen_streamlines, close
