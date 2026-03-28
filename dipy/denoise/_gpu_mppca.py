"""GPU-accelerated MPPCA denoising via WebGPU (wgpu-py).

Pipeline
--------
1. **extract_covariance** -- extract patch data, compute means and the
   D x D covariance matrix for each patch.
2. **jacobi_eigh** -- eigendecompose every covariance matrix with cyclic
   Jacobi iteration (multi-dispatch until converged).
3. **classify_reconstruct** -- run the Marcenko-Pastur classifier to
   separate signal from noise eigenvalues, zero noise columns, and
   reconstruct denoised patch data.
4. **accumulate** -- scatter-add weighted reconstructed voxels and
   weights (theta) back into the output volume using atomic floats
   (native f32 atomics when available, CAS fallback otherwise).

All GPU arithmetic is in float32.
"""

import importlib.resources
import logging
import os
import struct
import sys
import warnings

import numpy as np

logger = logging.getLogger(__name__)

_WGPU_STDERR_NOISE = (
    "no windowing system",
    "no config found",
    "max vertex attribute stride unknown",
    "using surfaceless platform",
)


def _call_suppressing_native_noise(fn, *args, **kwargs):
    import tempfile

    tmp, saved = tempfile.TemporaryFile(mode="w+"), os.dup(2)
    os.dup2(tmp.fileno(), 2)
    try:
        result = fn(*args, **kwargs)
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    tmp.seek(0)
    for ln in tmp:
        if not any(p in ln.lower() for p in _WGPU_STDERR_NOISE):
            sys.stderr.write(ln)
    tmp.close()
    return result


_WG_GRANULARITY = 32
_MAX_TILE_SIZE = 8192
_MAX_JACOBI_SWEEPS = 15
_MAX_REDUCE_BUF = 256
_JACOBI_EPSILON = 1e-5


def mppca_gpu(
    arr,
    *,
    mask=None,
    patch_radius=2,
    return_sigma=False,
    out_dtype=None,
    suppress_warning=False,
):
    """GPU-accelerated MPPCA denoising via WebGPU.

    Parameters
    ----------
    arr : ndarray, shape (Nx, Ny, Nz, D)
        4-D diffusion data.
    mask : ndarray, shape (Nx, Ny, Nz), optional
        Boolean brain mask.  ``True`` inside the region of interest.
    patch_radius : int or array-like of length 3, optional
        Half-width of the local patch (default 2 -> 5x5x5 patches).
    return_sigma : bool, optional
        If ``True`` also return the per-voxel noise-sigma estimate.
    out_dtype : dtype, optional
        Output dtype.  Defaults to the input dtype.
    suppress_warning : bool, optional
        If ``True``, suppress the dimensionality warning when
        ``patch_size < arr.shape[-1]``.

    Returns
    -------
    denoised : ndarray, shape (Nx, Ny, Nz, D)
    sigma : ndarray, shape (Nx, Ny, Nz)
        Only returned when *return_sigma* is ``True``.
    """
    try:
        import wgpu
    except ImportError as err:
        raise ImportError(
            "wgpu-py is required for backend='gpu'. Install with: pip install dipy[gpu]"
        ) from err

    adapter = _call_suppressing_native_noise(
        wgpu.gpu.request_adapter_sync, power_preference="high-performance"
    )

    if adapter is None:
        warnings.warn(
            "No WebGPU adapter found -- falling back to CPU MPPCA.",
            stacklevel=2,
        )
        from dipy.denoise.localpca import mppca

        return mppca(
            arr,
            mask=mask,
            patch_radius=patch_radius,
            return_sigma=return_sigma,
            out_dtype=out_dtype,
            backend="cpu",
        )

    use_f32_atomic = "shader-float32-atomic" in adapter.features
    required_features = []
    if use_f32_atomic:
        required_features.append("shader-float32-atomic")
        logger.info("GPU supports shader-float32-atomic; using native f32 atomics")
    else:
        logger.info("shader-float32-atomic not available; using CAS fallback")
    device = _call_suppressing_native_noise(
        adapter.request_device_sync, required_features=required_features
    )
    return _run_gpu_pipeline(
        device,
        arr,
        mask,
        patch_radius,
        return_sigma,
        out_dtype,
        suppress_warning,
        use_f32_atomic=use_f32_atomic,
    )


def _choose_tile_size(n_patches, D, device_limits):
    """Power-of-2 tile size fitting GPU buffer limits (>= 64)."""
    max_fit = device_limits["max-buffer-size"] // (4 * D * D * 4)
    T = max(min(n_patches, max_fit, _MAX_TILE_SIZE), 64)
    T = 1 << (T.bit_length() - 1)
    return T


def _precompute_patch_indices(shape, patch_radius, mask):
    """Build an (P, 6) uint32 array of patch bounding boxes.

    Parameters
    ----------
    shape : tuple
        ``(Nx, Ny, Nz, D)`` -- full array shape.
    patch_radius : ndarray, shape (3,)
        Patch half-widths along each spatial axis.
    mask : ndarray, shape (Nx, Ny, Nz)
        Boolean brain mask.

    Returns
    -------
    indices : ndarray, shape (P, 6), dtype uint32
        Each row is ``(ix1, ix2, jx1, jx2, kx1, kx2)`` where the
        half-open ranges ``[ix1, ix2)`` etc. define the patch volume.
    """
    pr0, pr1, pr2 = int(patch_radius[0]), int(patch_radius[1]), int(patch_radius[2])
    Nx, Ny, Nz = shape[0], shape[1], shape[2]

    k, j, i = np.nonzero(
        mask[pr0 : Nx - pr0, pr1 : Ny - pr1, pr2 : Nz - pr2].transpose(2, 1, 0)
    )

    patches = np.empty((len(i), 6), dtype=np.uint32)
    patches[:, 0] = i
    patches[:, 1] = i + 2 * pr0 + 1
    patches[:, 2] = j
    patches[:, 3] = j + 2 * pr1 + 1
    patches[:, 4] = k
    patches[:, 5] = k + 2 * pr2 + 1

    return patches


def _load_shader(name, D=None, B_MAX=None, use_global_mem=False, use_f32_atomic=False):
    """Read a ``.wgsl`` shader and render conditional blocks.

    Parameters
    ----------
    name : str
        Shader filename.
    D : int, optional
        Number of directions.  Sizes shared-memory arrays.
    B_MAX : int, optional
        Maximum tile size for covariance shared-memory tiling.  Only used
        for ``extract_covariance.wgsl``.
    use_global_mem : bool
        If True, render the global-memory variant; otherwise render the
        shared-memory variant.
    use_f32_atomic : bool
        If True, render the native f32 atomic variant; otherwise render
        the CAS (compare-and-swap) fallback variant.
    """
    import re

    pkg = importlib.resources.files("dipy.denoise._gpu_shaders")
    ref = pkg.joinpath(name)
    code = ref.read_text()

    _SHARED_ARRAY_PLACEHOLDER = "array<f32, 4096>"
    _COVARIANCE_MEANS_PLACEHOLDER = "array<f32, 512>"
    _COVARIANCE_PATCH_PLACEHOLDER = "array<f32, 123072>"

    if D is not None and not use_global_mem:
        if _SHARED_ARRAY_PLACEHOLDER in code:
            code = code.replace(_SHARED_ARRAY_PLACEHOLDER, f"array<f32, {D * (D + 1)}>")
        if _COVARIANCE_MEANS_PLACEHOLDER in code:
            code = code.replace(_COVARIANCE_MEANS_PLACEHOLDER, f"array<f32, {D}>")
        if B_MAX is not None and _COVARIANCE_PATCH_PLACEHOLDER in code:
            code = code.replace(
                _COVARIANCE_PATCH_PLACEHOLDER, f"array<f32, {B_MAX * D}>"
            )

    for keep, strip in [
        ("shared", "global") if not use_global_mem else ("global", "shared"),
        ("native", "cas") if use_f32_atomic else ("cas", "native"),
    ]:
        code = re.sub(
            rf"\{{\{{#{strip}\}}\}}.*?\{{\{{/{strip}\}}\}}\n?",
            "",
            code,
            flags=re.DOTALL,
        )
        code = re.sub(rf"\{{\{{[#/]{keep}\}}\}}\n?", "", code)

    return code


def _create_buffers(
    device, arr_f32, n_voxels, T, D, num_samples, patch_indices, return_sigma=False
):
    """Allocate all GPU buffers and upload initial data.

    Parameters
    ----------
    device : wgpu.GPUDevice
    arr_f32 : ndarray, shape (Nx, Ny, Nz, D)
        Input volume as contiguous float32.
    n_voxels : int
        ``Nx * Ny * Nz``.
    T : int
        Tile size (power of two).
    D : int
        Number of diffusion directions.
    num_samples : int
        Voxels per patch.
    patch_indices : ndarray, shape (P, 6), dtype uint32
    return_sigma : bool
        If True, allocate a staging buffer for sigma readback.

    Returns
    -------
    bufs : dict
        GPU buffers keyed by name.
    """
    import wgpu

    _BU = wgpu.BufferUsage
    STORAGE = _BU.STORAGE
    COPY_SRC = _BU.COPY_SRC
    COPY_DST = _BU.COPY_DST
    UNIFORM = _BU.UNIFORM
    MAP_READ = _BU.MAP_READ

    patch_bytes = patch_indices.tobytes()

    # fmt: off
    specs = [
        ("input",              arr_f32.nbytes,          STORAGE | COPY_DST),
        ("output",             n_voxels * D * 4,        STORAGE | COPY_SRC),
        ("theta",              n_voxels * 4,            STORAGE | COPY_SRC),
        ("sigma_out",          n_voxels * 4,            STORAGE | COPY_SRC),
        ("patch_indices_full", len(patch_bytes),         STORAGE | COPY_DST),
        ("staging_output",     n_voxels * D * 4,        COPY_DST | MAP_READ),
        ("staging_theta",      n_voxels * 4,            COPY_DST | MAP_READ),
        ("cov",                T * D * D * 4,           STORAGE),
        ("means",              T * D * 4,               STORAGE),
        ("eigenvectors",       T * D * D * 4,           STORAGE),
        ("eigenvalues",        T * D * 4,               STORAGE),
        ("converged",          T * 4,                   STORAGE | COPY_SRC | COPY_DST),
        ("off_diag",           T * 2 * 4,               STORAGE | COPY_DST),
        ("reconstructed",      T * num_samples * D * 4, STORAGE),
        ("ncomps",             T * 4,                   STORAGE),
        ("sigma_est",          T * 4,                   STORAGE),
        ("uniform_volume",     32,                      UNIFORM | COPY_DST),
        ("uniform_jacobi",     16,                      UNIFORM | COPY_DST),
    ]
    # fmt: on
    if return_sigma:
        specs.insert(7, ("staging_sigma", n_voxels * 4, COPY_DST | MAP_READ))

    bufs = {n: device.create_buffer(size=s, usage=u) for n, s, u in specs}

    device.queue.write_buffer(bufs["input"], 0, arr_f32)
    device.queue.write_buffer(bufs["patch_indices_full"], 0, patch_bytes)

    return bufs


def _create_pipelines(
    device, bufs, D, num_samples, use_global_mem, use_f32_atomic=False
):
    """Compile shaders, create compute pipelines and bind groups.

    Parameters
    ----------
    device : wgpu.GPUDevice
    bufs : dict
        GPU buffers returned by :func:`_create_buffers`.
    D : int
        Number of diffusion directions.
    num_samples : int
        Voxels per patch.
    use_global_mem : bool
        Whether to use global-memory shader variants.
    use_f32_atomic : bool
        If True, use native f32 atomics instead of CAS fallback.

    Returns
    -------
    pipes : dict
        ``{name: (pipeline, bind_group)}`` for each of the four passes.
    """
    limits = device.limits
    max_wg = limits["max-compute-workgroup-size-x"]
    max_shared = limits["max-compute-workgroup-storage-size"]

    _W = _WG_GRANULARITY
    # Pass 1, 3: one workgroup per patch, threads stripe over D.
    W13 = min(((D + _W - 1) // _W) * _W, max_wg)
    W13 = max(W13, _W)
    # Pass 2: Jacobi -- the shader uses a reduce_buf whose size caps the
    # workgroup.
    W2 = min(W13, _MAX_REDUCE_BUF)
    # Pass 4: threads stripe over voxels in the patch.
    W4 = min(((num_samples + _W - 1) // _W) * _W, max_wg)
    W4 = max(W4, _W)

    b_max = (max_shared - D * 4) // (D * 4)
    b_max = max(1, min(b_max, num_samples))
    shader_cov = device.create_shader_module(
        code=_load_shader("extract_covariance.wgsl", D=D, B_MAX=b_max),
    )
    shader_jacobi = device.create_shader_module(
        code=_load_shader("jacobi_eigh.wgsl", D=D, use_global_mem=use_global_mem),
    )
    shader_classify = device.create_shader_module(
        code=_load_shader(
            "classify_reconstruct.wgsl", D=D, use_global_mem=use_global_mem
        ),
    )
    shader_accum = device.create_shader_module(
        code=_load_shader("accumulate.wgsl", use_f32_atomic=use_f32_atomic),
    )

    pipe_specs = [
        ("cov", shader_cov, {"workgroup_size": int(W13), "b_max": int(b_max)}),
        ("jacobi", shader_jacobi, {"workgroup_size": int(W2)}),
        ("classify", shader_classify, {"workgroup_size": int(W13)}),
        ("accum", shader_accum, {"workgroup_size": int(W4)}),
    ]
    pipelines = {}
    for name, module, constants in pipe_specs:
        pipelines[name] = device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "main", "constants": constants},
        )

    bind_specs = {
        "cov": ["uniform_volume", "input", "patch_indices_full", "cov", "means"],
        "jacobi": [
            "uniform_jacobi",
            "cov",
            "eigenvectors",
            "eigenvalues",
            "converged",
            "off_diag",
        ],
        "classify": [
            "uniform_volume",
            "input",
            "patch_indices_full",
            "eigenvalues",
            "eigenvectors",
            "means",
            "reconstructed",
            "ncomps",
            "sigma_est",
        ],
        "accum": [
            "uniform_volume",
            "patch_indices_full",
            "reconstructed",
            "ncomps",
            "sigma_est",
            "output",
            "theta",
            "sigma_out",
        ],
    }

    return {
        name: (
            pipelines[name],
            device.create_bind_group(
                layout=pipelines[name].get_bind_group_layout(0),
                entries=[
                    {"binding": i, "resource": {"buffer": bufs[b]}}
                    for i, b in enumerate(buf_names)
                ],
            ),
        )
        for name, buf_names in bind_specs.items()
    }


def _read_staging_f32(buf, shape):
    """Read a mapped staging buffer as float32 and unmap it."""
    out = np.frombuffer(buf.read_mapped(), np.float32).reshape(shape).copy()
    buf.unmap()
    return out


def _map_staging_buffers(device, staging_bufs):
    """Map staging buffers for CPU readback."""
    try:
        awaitables = [b._map("READ_NOSYNC") for b in staging_bufs]
        device._poll_wait()
        for a in awaitables:
            a._finish()
    except AttributeError:
        for b in staging_bufs:
            b.map("READ")


def _readback_results(device, bufs, mask, vol_shape, return_sigma, out_dtype):
    """Copy GPU results to CPU, compute weighted average, and apply mask.

    Parameters
    ----------
    device : wgpu.GPUDevice
    bufs : dict
        GPU buffers returned by :func:`_create_buffers`.
    mask : ndarray, shape (Nx, Ny, Nz)
    vol_shape : tuple
        ``(Nx, Ny, Nz, D)`` -- full volume shape.
    return_sigma : bool
    out_dtype : dtype

    Returns
    -------
    result : ndarray, shape (Nx, Ny, Nz, D)
    sigma : ndarray, shape (Nx, Ny, Nz)
        Only returned when *return_sigma* is ``True``.
    """
    Nx, Ny, Nz, D = vol_shape
    n_voxels = Nx * Ny * Nz
    enc = device.create_command_encoder()
    enc.copy_buffer_to_buffer(
        bufs["output"], 0, bufs["staging_output"], 0, n_voxels * D * 4
    )
    enc.copy_buffer_to_buffer(bufs["theta"], 0, bufs["staging_theta"], 0, n_voxels * 4)
    if return_sigma:
        enc.copy_buffer_to_buffer(
            bufs["sigma_out"], 0, bufs["staging_sigma"], 0, n_voxels * 4
        )
    device.queue.submit([enc.finish()])

    staging_bufs = [bufs["staging_output"], bufs["staging_theta"]]
    if return_sigma:
        staging_bufs.append(bufs["staging_sigma"])

    _map_staging_buffers(device, staging_bufs)

    output_f32 = _read_staging_f32(bufs["staging_output"], (Nx, Ny, Nz, D))
    theta_f32 = _read_staging_f32(bufs["staging_theta"], (Nx, Ny, Nz))

    nonzero = theta_f32 > 0.0
    output_f32[nonzero] /= theta_f32[nonzero, np.newaxis]
    output_f32[~mask] = 0.0
    np.clip(output_f32, 0.0, None, out=output_f32)

    result = output_f32.astype(out_dtype)

    if return_sigma:
        sigma_f32 = _read_staging_f32(bufs["staging_sigma"], (Nx, Ny, Nz))
        sigma_f32[nonzero] /= theta_f32[nonzero]
        sigma_f32[~mask] = 0.0
        return result, np.sqrt(sigma_f32).astype(out_dtype)

    return result


def _run_gpu_pipeline(
    device,
    arr,
    mask,
    patch_radius,
    return_sigma,
    out_dtype,
    suppress_warning=False,
    use_f32_atomic=False,
):
    """Execute the 4-pass GPU pipeline and return the denoised volume.

    Parameters
    ----------
    device : wgpu.GPUDevice
    arr : ndarray, shape (Nx, Ny, Nz, D)
    mask : ndarray or None
    patch_radius : int or array-like
    return_sigma : bool
    out_dtype : dtype or None
    """
    if arr.ndim != 4:
        raise ValueError(f"GPU MPPCA requires a 4-D array, got shape {arr.shape}.")

    if out_dtype is None:
        out_dtype = arr.dtype

    if arr.dtype != np.float32:
        warnings.warn(
            f"GPU MPPCA converts input from {arr.dtype} to float32. "
            "Results may differ from the CPU path at full precision.",
            stacklevel=3,
        )
    arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    Nx, Ny, Nz, D = arr_f32.shape

    if mask is None:
        mask = np.ones((Nx, Ny, Nz), dtype=bool)

    from dipy.denoise.localpca import (
        compute_num_samples,
        compute_patch_size,
        create_patch_radius_arr,
        dimensionality_problem_message,
    )

    pr = create_patch_radius_arr(arr, patch_radius)
    patch_size = compute_patch_size(pr)
    num_samples = compute_num_samples(patch_size)
    if (num_samples - 1) < D and not suppress_warning:
        spr = max(np.max(pr) + 1, 2)
        warnings.warn(
            dimensionality_problem_message(arr, num_samples, spr),
            stacklevel=2,
        )
    if num_samples <= 1:
        raise ValueError("Effective patch has only 1 sample -- increase patch_radius.")

    patch_indices = _precompute_patch_indices(arr.shape, pr, mask)
    n_patches = len(patch_indices)
    if n_patches == 0:
        result = np.zeros_like(arr_f32).astype(out_dtype)
        if return_sigma:
            return result, np.zeros((Nx, Ny, Nz), dtype=out_dtype)
        return result

    logger.info(
        "GPU MPPCA: %d patches, D=%d, num_samples=%d",
        n_patches,
        D,
        num_samples,
    )

    jacobi_shared = 2 * D * (D + 1) * 4
    max_shared = device.limits["max-compute-workgroup-storage-size"]
    use_global_mem = jacobi_shared > max_shared
    if use_global_mem:
        logger.info(
            "D=%d exceeds shared memory (%d > %d bytes), using global-memory shaders",
            D,
            jacobi_shared,
            max_shared,
        )

    T = _choose_tile_size(n_patches, D, device.limits)
    logger.info("Tile size: T=%d", T)

    n_voxels = Nx * Ny * Nz
    bufs = _create_buffers(
        device,
        arr_f32,
        n_voxels,
        T,
        D,
        num_samples,
        patch_indices,
        return_sigma=return_sigma,
    )
    pipes = _create_pipelines(
        device,
        bufs,
        D,
        num_samples,
        use_global_mem,
        use_f32_atomic=use_f32_atomic,
    )

    try:
        from tqdm import tqdm

        tile_iter = tqdm(
            range(0, n_patches, T),
            desc="GPU MPPCA",
            unit="tile",
        )
    except ImportError:
        tile_iter = range(0, n_patches, T)

    pipe_cov, bg_cov = pipes["cov"]
    pipe_jacobi, bg_jacobi = pipes["jacobi"]
    pipe_classify, bg_classify = pipes["classify"]
    pipe_accum, bg_accum = pipes["accum"]
    buf_uniform_volume = bufs["uniform_volume"]
    buf_uniform_jacobi = bufs["uniform_jacobi"]

    for tile_start in tile_iter:
        tile_end = min(tile_start + T, n_patches)
        current_T = tile_end - tile_start

        device.queue.write_buffer(
            buf_uniform_volume,
            0,
            struct.pack("<7I1I", D, Nx, Ny, Nz, num_samples, tile_start, current_T, 0),
        )
        device.queue.write_buffer(
            buf_uniform_jacobi,
            0,
            struct.pack("<2IfI", D, current_T, _JACOBI_EPSILON, 0),
        )
        device.queue.write_buffer(bufs["converged"], 0, bytes(current_T * 4))
        device.queue.write_buffer(bufs["off_diag"], 0, bytes(current_T * 2 * 4))

        # wgpu-native inserts implicit storage barriers between dispatches
        # within a single compute pass. Browser WebGPU does not guarantee
        # this; porting would require separate compute passes per stage.
        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()

        cp.set_pipeline(pipe_cov)
        cp.set_bind_group(0, bg_cov)
        cp.dispatch_workgroups(current_T)

        for _ in range(_MAX_JACOBI_SWEEPS):
            cp.set_pipeline(pipe_jacobi)
            cp.set_bind_group(0, bg_jacobi)
            cp.dispatch_workgroups(current_T)

        cp.set_pipeline(pipe_classify)
        cp.set_bind_group(0, bg_classify)
        cp.dispatch_workgroups(current_T)

        cp.set_pipeline(pipe_accum)
        cp.set_bind_group(0, bg_accum)
        cp.dispatch_workgroups(current_T)

        cp.end()
        device.queue.submit([enc.finish()])

    return _readback_results(
        device, bufs, mask, (Nx, Ny, Nz, D), return_sigma, out_dtype
    )
