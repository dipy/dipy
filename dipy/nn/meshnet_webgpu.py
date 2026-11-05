"""GPU inference of MeshNet models through WebGPU.

WebGPU runs on Vulkan (Linux, Windows), Metal (macOS) and Direct3D 12
(Windows), so any recent GPU can be used without CUDA. The compute shaders
follow the design of the WebGPU backend of brainchopC (MIT license, see
meshnet_cpu.c): channel-last activations padded to 16 channels, each thread
computing all channels of two voxels, and a two-stage reduction for the group
normalization.
"""

import struct

import numpy as np

from dipy.nn.meshnet import pack_meshnet_kernels
from dipy.utils.optpkg import optional_package

wgpu, have_wgpu, _ = optional_package("wgpu", min_version="0.31.0")

STRIDE = 16
VOXELS_PER_THREAD = 2
WORKGROUP_SIZE = 64
REDUCE_SIZE = 256

_GELU = """
fn mn_tanh(x_in : f32) -> f32 {
  let x = clamp(x_in, -9.0, 9.0);
  let u = x * x;
  var p = -8.29118133e-14;
  p = p * u + 5.19263868e-11;
  p = p * u - 2.00294448e-08;
  p = p * u + 1.11017944e-05;
  p = p * u + 0.00309865153;
  p = p * u + 0.130791619;
  p = p * u + 0.99999994;
  var q = 0.000253859733;
  q = q * u + 0.024473751;
  q = q * u + 0.464124829;
  q = q * u + 1.0;
  return x * p / q;
}
fn gelu(x : f32) -> f32 {
  let u = 0.797884583 * (x + 0.044715 * (x * x * x));
  return (0.5 * x) * (1.0 + mn_tanh(u));
}
"""


def _preamble(shape, *, storage):
    nz, ny, nx = shape
    enable = "enable f16;\n" if storage == "f16" else ""
    return (
        f"{enable}alias T = {storage};\n"
        f"const NX : i32 = {nx};\nconst NY : i32 = {ny};\nconst NZ : i32 = {nz};\n"
        f"const CS : u32 = {STRIDE}u;\n"
    )


def _conv_source(shape, *, storage, first):
    """WGSL of a zero-padded dilated 3x3x3 convolution.

    Each thread accumulates the 16 output channels of VOXELS_PER_THREAD
    consecutive voxels in named vec4 registers, so that every weight vector is
    loaded once for all of them.
    """
    voxels = range(VOXELS_PER_THREAD)
    if first:
        source = "@group(0) @binding(0) var<storage, read> src : array<T>;"
    else:
        source = "@group(0) @binding(0) var<storage, read> src4 : array<vec4<T>>;"
    lines = [
        _preamble(shape, storage=storage),
        "struct Params { dilation : i32, weight_offset : u32 };",
        source,
        "@group(0) @binding(1) var<storage, read_write> dst : array<T>;",
        "@group(0) @binding(2) var<uniform> params : Params;",
        "@group(0) @binding(3) var<storage, read> wts4 : array<vec4<T>>;",
        f"@compute @workgroup_size({WORKGROUP_SIZE}, 1, 1)",
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {",
        "  let y = i32(gid.y);",
        "  let z = i32(gid.z);",
        f"  let x0 = i32(gid.x) * {VOXELS_PER_THREAD};",
        "  if (x0 >= NX) { return; }",
        "  let d = params.dilation;",
        "  let wq0 = params.weight_offset / 4u;",
    ]
    lines += [f"  var a{v}{n} = vec4<f32>(0.0);" for v in voxels for n in range(4)]
    lines += [
        "  var tap : u32 = 0u;",
        "  for (var dz : i32 = -1; dz <= 1; dz++) {",
        "  for (var dy : i32 = -1; dy <= 1; dy++) {",
        "    let sz = z + dz * d;",
        "    let sy = y + dy * d;",
        "    let okyz = sz >= 0 && sz < NZ && sy >= 0 && sy < NY;",
        "    let plane = (u32(clamp(sz, 0, NZ - 1)) * u32(NY) +",
        "                 u32(clamp(sy, 0, NY - 1))) * u32(NX);",
        "  for (var dx : i32 = -1; dx <= 1; dx++) {",
        "    if (okyz) {",
    ]
    if first:
        lines.append("      let wq = wq0 + tap * 4u;")
        lines += [f"      let w{n} = vec4<f32>(wts4[wq + {n}u]);" for n in range(4)]
        for v in voxels:
            lines += [
                "      {",
                f"        let sx = x0 + {v} + dx * d;",
                "        let sv = f32(src[plane + u32(clamp(sx, 0, NX - 1))]);",
                "        let s = vec4<f32>(select(0.0, sv, sx >= 0 && sx < NX));",
            ]
            lines += [f"        a{v}{n} = fma(w{n}, s, a{v}{n});" for n in range(4)]
            lines.append("      }")
    else:
        lines.append("      for (var ic4 : u32 = 0u; ic4 < 4u; ic4++) {")
        for v in voxels:
            lines += [
                f"        let sx{v} = x0 + {v} + dx * d;",
                f"        let q{v} = vec4<f32>(src4[(plane + u32(clamp(sx{v}, 0,"
                " NX - 1))) * 4u + ic4]);",
                f"        let s{v} = select(vec4<f32>(0.0), q{v}, sx{v} >= 0 &&"
                f" sx{v} < NX);",
            ]
        for i, component in enumerate("xyzw"):
            lines += [
                "        {",
                f"          let wq = wq0 + (tap * 16u + ic4 * 4u + {i}u) * 4u;",
            ]
            lines += [
                f"          let w{n} = vec4<f32>(wts4[wq + {n}u]);" for n in range(4)
            ]
            lines += [
                f"          a{v}{n} = fma(w{n}, vec4<f32>(s{v}.{component}), a{v}{n});"
                for v in voxels
                for n in range(4)
            ]
            lines.append("        }")
        lines.append("      }")
    lines += ["    }", "    tap++;", "  }}}"]
    for v in voxels:
        lines += [
            f"  if (x0 + {v} < NX) {{",
            "    let o = ((u32(z) * u32(NY) + u32(y)) * u32(NX) +"
            f" u32(x0 + {v})) * CS;",
        ]
        lines += [
            f"    dst[o + {4 * n + k}u] = T(a{v}{n}.{component});"
            for n in range(4)
            for k, component in enumerate("xyzw")
        ]
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _moments_source(shape, *, storage):
    """WGSL of the per-workgroup channel sums and sums of squares."""
    size = WORKGROUP_SIZE
    return (
        _preamble(shape, storage=storage)
        + f"""
@group(0) @binding(0) var<storage, read> act : array<T>;
@group(0) @binding(1) var<storage, read_write> mom : array<f32>;
var<workgroup> wg_sum : array<f32, {size * STRIDE}>;
var<workgroup> wg_sq : array<f32, {size * STRIDE}>;
@compute @workgroup_size({size}, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_index) lid : u32,
        @builtin(workgroup_id) wid : vec3<u32>,
        @builtin(num_workgroups) nwg : vec3<u32>) {{
  let slot = lid * CS;
  for (var c : u32 = 0u; c < CS; c++) {{
    wg_sum[slot + c] = 0.0;
    wg_sq[slot + c] = 0.0;
  }}
  if (gid.x < u32(NX)) {{
    let base = ((gid.z * u32(NY) + gid.y) * u32(NX) + gid.x) * CS;
    for (var c : u32 = 0u; c < CS; c++) {{
      let v = f32(act[base + c]);
      wg_sum[slot + c] = v;
      wg_sq[slot + c] = v * v;
    }}
  }}
  workgroupBarrier();
  if (lid < CS) {{
    var s : f32 = 0.0;
    var q : f32 = 0.0;
    for (var l : u32 = 0u; l < {size}u; l++) {{
      s += wg_sum[l * CS + lid];
      q += wg_sq[l * CS + lid];
    }}
    let g = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    mom[(g * CS + lid) * 2u] = s;
    mom[(g * CS + lid) * 2u + 1u] = q;
  }}
}}
"""
    )


def _finish_source(shape, *, storage, groups):
    """WGSL folding the partial sums into a mean and inverse std per channel.

    The partials are folded in independent stripes before being combined, as
    a single f32 running sum over hundreds of thousands of terms would lose
    the small ones.
    """
    n_voxels = float(np.prod(shape))
    stripes = REDUCE_SIZE // STRIDE
    stats = groups * STRIDE * 2
    return (
        _preamble(shape, storage=storage)
        + f"""
@group(0) @binding(0) var<storage, read_write> mom : array<f32>;
var<workgroup> fs : array<f32, {REDUCE_SIZE}>;
var<workgroup> fq : array<f32, {REDUCE_SIZE}>;
@compute @workgroup_size({REDUCE_SIZE}, 1, 1)
fn main(@builtin(local_invocation_index) lid : u32) {{
  let channel = lid % CS;
  var s : f32 = 0.0;
  var q : f32 = 0.0;
  for (var g : u32 = lid / CS; g < {groups}u; g += {stripes}u) {{
    s += mom[(g * CS + channel) * 2u];
    q += mom[(g * CS + channel) * 2u + 1u];
  }}
  fs[lid] = s;
  fq[lid] = q;
  workgroupBarrier();
  if (lid < CS) {{
    var ts : f32 = 0.0;
    var tq : f32 = 0.0;
    for (var k : u32 = 0u; k < {stripes}u; k++) {{
      ts += fs[k * CS + lid];
      tq += fq[k * CS + lid];
    }}
    let mean = ts / {n_voxels};
    let variance = max(tq / {n_voxels} - mean * mean, 0.0);
    mom[{stats}u + lid * 2u] = mean;
    mom[{stats}u + lid * 2u + 1u] = inverseSqrt(variance + 1e-5);
  }}
}}
"""
    )


def _norm_source(shape, *, storage, groups):
    """WGSL of the in-place group normalization and GELU."""
    stats = groups * STRIDE * 2
    return (
        _preamble(shape, storage=storage)
        + _GELU
        + f"""
@group(0) @binding(0) var<storage, read_write> act : array<T>;
@group(0) @binding(1) var<storage, read> mom : array<f32>;
@compute @workgroup_size({WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {{
  if (gid.x >= u32(NX)) {{ return; }}
  let base = ((gid.z * u32(NY) + gid.y) * u32(NX) + gid.x) * CS;
  for (var c : u32 = 0u; c < CS; c++) {{
    let mean = mom[{stats}u + c * 2u];
    let inv = mom[{stats}u + c * 2u + 1u];
    act[base + c] = T(gelu((f32(act[base + c]) - mean) * inv));
  }}
}}
"""
    )


def _classify_source(shape, *, storage, channels, n_classes):
    """WGSL of the final 1x1x1 convolution, writing class-major logits."""
    n_voxels = int(np.prod(shape))
    return (
        _preamble(shape, storage=storage)
        + f"""
@group(0) @binding(0) var<storage, read> act : array<T>;
@group(0) @binding(1) var<storage, read_write> logits : array<f32>;
@group(0) @binding(2) var<storage, read> cls : array<f32>;
@compute @workgroup_size({WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {{
  if (gid.x >= u32(NX)) {{ return; }}
  let i = (gid.z * u32(NY) + gid.y) * u32(NX) + gid.x;
  for (var k : u32 = 0u; k < {n_classes}u; k++) {{
    var acc = cls[{channels * n_classes}u + k];
    for (var c : u32 = 0u; c < {channels}u; c++) {{
      acc += f32(act[i * CS + c]) * cls[c * {n_classes}u + k];
    }}
    logits[k * {n_voxels}u + i] = acc;
  }}
}}
"""
    )


def webgpu_device(*, allow_software=False):
    """Request a WebGPU device on the most powerful GPU available.

    Half precision is enabled on the device when the GPU supports it.

    Parameters
    ----------
    allow_software : bool, optional
        Whether to accept a software implementation running on the CPU, such
        as Mesa's lavapipe. These are much slower than the CPU kernels of
        ``dipy.nn.meshnet`` and only useful for testing.

    Returns
    -------
    wgpu.GPUDevice or None
        The device, or None if ``wgpu`` is not installed or no suitable
        adapter was found.
    """
    if not have_wgpu:
        return None
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            return None
        if adapter.info.get("adapter_type") == "CPU" and not allow_software:
            return None
        features = [f for f in ("shader-f16",) if f in adapter.features]
        limits = {
            name: adapter.limits[name]
            for name in ("max-storage-buffer-binding-size", "max-buffer-size")
        }
        return adapter.request_device_sync(
            required_features=features, required_limits=limits
        )
    except (RuntimeError, wgpu.GPUError):
        return None


def meshnet_forward_webgpu(
    volume,
    kernels,
    dilations,
    classifier,
    bias,
    *,
    device,
    half_precision=False,
):
    """Compute the class logits of a MeshNet model on a WebGPU device.

    Same computation as ``dipy.nn.meshnet.meshnet_forward_cpu``. Arithmetic
    is done in float32. With ``half_precision``, activations and weights are
    stored in float16, halving the GPU memory needed.

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
    device : wgpu.GPUDevice
        Device to run on, see ``webgpu_device``.
    half_precision : bool, optional
        Whether to store activations and weights in float16.

    Returns
    -------
    np.ndarray (n_classes, X, Y, Z)
        Class logits, as float32.

    Raises
    ------
    MemoryError
        If the activations do not fit in a buffer of the device.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"volume should be 3D, got {volume.ndim}D")
    kernels = [np.asarray(kernel, dtype=np.float32) for kernel in kernels]
    if len(kernels) == 0 or len(dilations) != len(kernels):
        raise ValueError("kernels and dilations should have the same length")
    channels = kernels[0].shape[0]
    if not 1 <= channels <= STRIDE:
        raise ValueError(f"channels should be between 1 and {STRIDE}")
    if half_precision and "shader-f16" not in device.features:
        raise ValueError("This device does not support half precision")

    storage = "f16" if half_precision else "f32"
    dtype = np.float16 if half_precision else np.float32
    shape = volume.shape
    nz, ny, nx = shape
    n_voxels = int(np.prod(shape))
    n_classes = classifier.shape[0]
    activation_size = n_voxels * STRIDE * np.dtype(dtype).itemsize
    if activation_size > min(
        device.limits["max-storage-buffer-binding-size"],
        device.limits["max-buffer-size"],
    ):
        raise MemoryError(
            f"Activations of {activation_size / 2**30:.1f} GiB exceed the "
            "largest buffer of this WebGPU device"
        )

    groups = -(-nx // WORKGROUP_SIZE) * ny * nz
    conv_groups = (-(-nx // (WORKGROUP_SIZE * VOXELS_PER_THREAD)), ny, nz)
    voxel_groups = (-(-nx // WORKGROUP_SIZE), ny, nz)
    first, hidden = pack_meshnet_kernels(kernels, channels)
    weights = np.concatenate([first.ravel(), hidden.ravel()]).astype(dtype)
    head = np.concatenate([np.asarray(classifier).T.ravel(), np.asarray(bias)])

    storage_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    buffers = []

    def new_buffer(*, data=None, size=None, usage=storage_usage):
        if data is None:
            buffer = device.create_buffer(size=size, usage=usage)
        else:
            buffer = device.create_buffer_with_data(
                data=np.ascontiguousarray(data).tobytes(), usage=usage
            )
        buffers.append(buffer)
        return buffer

    def new_pipeline(source):
        module = device.create_shader_module(code=source)
        return device.create_compute_pipeline(
            layout="auto", compute={"module": module, "entry_point": "main"}
        )

    def new_group(pipeline, resources):
        return device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                for i, b in enumerate(resources)
            ],
        )

    try:
        source = new_buffer(data=volume.astype(dtype))
        activations = [new_buffer(size=activation_size) for _ in range(2)]
        weight_buffer = new_buffer(data=weights)
        moments = new_buffer(size=(groups + 1) * STRIDE * 2 * 4)
        head_buffer = new_buffer(data=head.astype(np.float32))
        logits = new_buffer(
            size=n_classes * n_voxels * 4,
            usage=storage_usage | wgpu.BufferUsage.COPY_SRC,
        )
        offsets = np.cumsum([0, first.size] + [hidden[0].size] * (len(kernels) - 2))
        params = [
            new_buffer(
                data=np.frombuffer(struct.pack("iIII", d, offset, 0, 0), np.uint8),
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            for d, offset in zip(dilations, offsets)
        ]

        conv_first = new_pipeline(_conv_source(shape, storage=storage, first=True))
        conv_hidden = new_pipeline(_conv_source(shape, storage=storage, first=False))
        stats = new_pipeline(_moments_source(shape, storage=storage))
        finish = new_pipeline(_finish_source(shape, storage=storage, groups=groups))
        norm = new_pipeline(_norm_source(shape, storage=storage, groups=groups))
        classify = new_pipeline(
            _classify_source(
                shape, storage=storage, channels=channels, n_classes=n_classes
            )
        )
        stats_groups = [new_group(stats, [a, moments]) for a in activations]
        finish_group = new_group(finish, [moments])
        norm_groups = [new_group(norm, [a, moments]) for a in activations]

        encoder = device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        current = 0
        for layer in range(len(kernels)):
            if layer == 0:
                compute.set_pipeline(conv_first)
                group = new_group(
                    conv_first, [source, activations[0], params[0], weight_buffer]
                )
            else:
                current = 1 - current
                compute.set_pipeline(conv_hidden)
                group = new_group(
                    conv_hidden,
                    [
                        activations[1 - current],
                        activations[current],
                        params[layer],
                        weight_buffer,
                    ],
                )
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups(*conv_groups)
            compute.set_pipeline(stats)
            compute.set_bind_group(0, stats_groups[current])
            compute.dispatch_workgroups(*voxel_groups)
            compute.set_pipeline(finish)
            compute.set_bind_group(0, finish_group)
            compute.dispatch_workgroups(1, 1, 1)
            compute.set_pipeline(norm)
            compute.set_bind_group(0, norm_groups[current])
            compute.dispatch_workgroups(*voxel_groups)
        compute.set_pipeline(classify)
        compute.set_bind_group(
            0, new_group(classify, [activations[current], logits, head_buffer])
        )
        compute.dispatch_workgroups(*voxel_groups)
        compute.end()
        device.queue.submit([encoder.finish()])
        data = device.queue.read_buffer(logits)
        return (
            np.frombuffer(data, dtype=np.float32).reshape((n_classes,) + shape).copy()
        )
    finally:
        for buffer in buffers:
            buffer.destroy()
