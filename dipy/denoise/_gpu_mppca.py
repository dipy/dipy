"""GPU-accelerated MPPCA denoising via WebGPU (wgpu-py).

This module implements the same Marcenko-Pastur PCA denoising as
:func:`dipy.denoise.localpca.genpca` with ``sigma=None`` and
``pca_method='eig'``, but offloads the heavy lifting to a 4-pass
compute-shader pipeline executed on the GPU.

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
   weights (theta) back into the output volume using atomic float CAS.

All GPU arithmetic is in float32. The input array is converted on upload
and the result is cast to *out_dtype* before return.

Tiling
------
Patches are processed in tiles of *T* (a power of two chosen to fit GPU
buffer limits). Per-tile buffers for covariance matrices, eigenvectors,
etc. are allocated once and reused across tiles.
"""

import importlib.resources
import logging
import struct
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WARP_SIZE = 32
_MAX_TILE_SIZE = 8192
_MAX_JACOBI_SWEEPS = 15
_MAX_REDUCE_BUF = 256
_JACOBI_EPSILON = 1e-5

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def mppca_gpu(arr, *, mask=None, patch_radius=2,
              return_sigma=False, out_dtype=None,
              suppress_warning=False):
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
        import wgpu  # noqa: F401
    except ImportError:
        raise ImportError(
            "wgpu-py is required for backend='gpu'. "
            "Install with: pip install dipy[gpu]"
        )

    adapter = wgpu.gpu.request_adapter_sync(
        power_preference="high-performance"
    )
    if adapter is None:
        warnings.warn(
            "No WebGPU adapter found -- falling back to CPU MPPCA.",
            stacklevel=2,
        )
        from dipy.denoise.localpca import mppca
        return mppca(
            arr, mask=mask, patch_radius=patch_radius,
            return_sigma=return_sigma, out_dtype=out_dtype, backend="cpu",
        )

    device = adapter.request_device_sync()
    return _run_gpu_pipeline(
        device, arr, mask, patch_radius, return_sigma, out_dtype,
        suppress_warning,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _choose_tile_size(n_patches, D, device_limits):
    """Pick a power-of-2 tile size that fits in the GPU buffer limit.

    The dominant per-tile allocation is ``T * D * D * 4`` bytes for the
    covariance, eigenvector, and cov-copy matrices (three such buffers).
    We also need ``T * num_samples * D * 4`` for the reconstruction
    buffer, but that is bounded by the same order for typical D.

    Parameters
    ----------
    n_patches : int
    D : int
        Number of diffusion directions.
    device_limits : dict

    Returns
    -------
    T : int
        Tile size (power of two, >= 64).
    """
    max_buffer = device_limits["max-buffer-size"]
    # Three D*D matrices + eigenvalues + means + ncomps + sigma + converged
    # + off_diag_norms per patch.  Conservatively estimate 4 * D*D floats.
    per_patch_bytes = 4 * D * D * 4
    max_by_memory = max(max_buffer // per_patch_bytes, 64)
    T = min(n_patches, max_by_memory, _MAX_TILE_SIZE)
    T = max(T, 64)
    # Round down to power of two.
    T = 1 << (T.bit_length() - 1)
    return T


def _precompute_patch_indices(shape, patch_radius, mask):
    """Build an (P, 6) uint32 array of patch bounding boxes.

    Replicates the exact iteration order of
    :func:`dipy.denoise.localpca.genpca` (lines 315-326).

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

    # Pre-allocate a generous buffer; trim at the end.
    buf = np.empty(((Nx * Ny * Nz), 6), dtype=np.uint32)
    count = 0

    for k in range(pr2, Nz - pr2):
        for j in range(pr1, Ny - pr1):
            for i in range(pr0, Nx - pr0):
                if not mask[i, j, k]:
                    continue
                buf[count, 0] = i - pr0
                buf[count, 1] = i + pr0 + 1
                buf[count, 2] = j - pr1
                buf[count, 3] = j + pr1 + 1
                buf[count, 4] = k - pr2
                buf[count, 5] = k + pr2 + 1
                count += 1

    return buf[:count].copy()


def _load_shader(name, D=None):
    """Read a ``.wgsl`` shader and template shared-memory sizes for *D*.

    The Jacobi and classify shaders declare fixed-size workgroup arrays
    (``array<f32, 4096>`` for a 64×64 default).  When *D* is provided the
    source is patched so that every ``array<f32, 4096>`` becomes
    ``array<f32, D*D>`` and the ``d_max`` override is set to *D*.  This
    avoids both overflow (D > 64) and wasted shared memory (D < 64).
    """
    ref = importlib.resources.files(
        "dipy.denoise._gpu_shaders"
    ).joinpath(name)
    code = ref.read_text()
    if D is not None:
        code = code.replace("array<f32, 4096>", f"array<f32, {D * D}>")
    return code


# ---------------------------------------------------------------------------
# Uniform packing helpers
# ---------------------------------------------------------------------------
# WGSL structs have strict 16-byte alignment rules.  We match the layout
# of each shader's ``Params`` struct with explicit padding.


def _pack_volume_params(D, Nx, Ny, Nz, num_samples, tile_offset, tile_size):
    """Pack the uniform buffer for *extract_covariance*.

    WGSL layout::

        struct Params {
            D:            u32,  // offset 0
            Nx:           u32,  // offset 4
            Ny:           u32,  // offset 8
            Nz:           u32,  // offset 12
            num_samples:  u32,  // offset 16
            tile_offset:  u32,  // offset 20
            tile_size:    u32,  // offset 24
        }

    Total 28 bytes; padded to 32 (next multiple of 16).
    """
    return struct.pack("<7I1I", D, Nx, Ny, Nz, num_samples,
                       tile_offset, tile_size, 0)


def _pack_jacobi_params(D, tile_size, sweep_idx, epsilon):
    """Pack the uniform buffer for *jacobi_eigh*.

    WGSL layout::

        struct Params {
            D:         u32,   // offset 0
            tile_size: u32,   // offset 4
            sweep_idx: u32,   // offset 8
            epsilon:   f32,   // offset 12
        }

    Total 16 bytes -- already 16-byte aligned.
    """
    return struct.pack("<3If", D, tile_size, sweep_idx, epsilon)


def _pack_accumulate_params(D, Nx, Ny, Nz, num_samples, tile_size):
    """Pack the uniform buffer for *accumulate*.

    WGSL layout::

        struct Params {
            D:            u32,  // offset 0
            Nx:           u32,  // offset 4
            Ny:           u32,  // offset 8
            Nz:           u32,  // offset 12
            num_samples:  u32,  // offset 16
            tile_size:    u32,  // offset 20
        }

    Total 24 bytes; padded to 32.
    """
    return struct.pack("<6I2I", D, Nx, Ny, Nz, num_samples,
                       tile_size, 0, 0)


# ---------------------------------------------------------------------------
# Main GPU pipeline
# ---------------------------------------------------------------------------


def _run_gpu_pipeline(device, arr, mask, patch_radius, return_sigma,
                      out_dtype, suppress_warning=False):
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
    import wgpu

    # -- 0. Validate / prepare -----------------------------------------------
    if arr.ndim != 4:
        raise ValueError(
            f"GPU MPPCA requires a 4-D array, got shape {arr.shape}."
        )

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
        create_patch_radius_arr, compute_patch_size, compute_num_samples,
        dimensionality_problem_message,
    )
    pr = create_patch_radius_arr(arr, patch_radius)
    patch_size = compute_patch_size(pr)
    num_samples = compute_num_samples(patch_size)
    if (num_samples - 1) < D and not suppress_warning:
        spr = max(np.max(pr) + 1, 2)
        warnings.warn(dimensionality_problem_message(arr, num_samples, spr))
    if num_samples <= 1:
        raise ValueError(
            "Effective patch has only 1 sample -- increase patch_radius."
        )

    # -- 1. Build patch indices ----------------------------------------------
    patch_indices = _precompute_patch_indices(arr.shape, pr, mask)
    n_patches = len(patch_indices)
    if n_patches == 0:
        # Nothing inside the mask -- return zeros.
        result = np.zeros_like(arr_f32).astype(out_dtype)
        if return_sigma:
            return result, np.zeros((Nx, Ny, Nz), dtype=out_dtype)
        return result

    logger.info(
        "GPU MPPCA: %d patches, D=%d, num_samples=%d",
        n_patches, D, num_samples,
    )

    # -- 2. Shader variant selection ------------------------------------------
    # Jacobi needs 2 × D² × 4 bytes of shared memory (C + V matrices).
    # Classify needs 1 × D² × 4 bytes.  If D is too large for shared
    # memory, use global-memory shader variants (slower but no size limit).
    jacobi_shared = 2 * D * D * 4
    max_shared = device.limits["max-compute-workgroup-storage-size"]
    use_global_mem = jacobi_shared > max_shared
    if use_global_mem:
        logger.info(
            "D=%d exceeds shared memory (%d > %d bytes), using global-memory shaders",
            D, jacobi_shared, max_shared,
        )

    # -- 3. Tile size --------------------------------------------------------
    T = _choose_tile_size(n_patches, D, device.limits)
    logger.info("Tile size: T=%d", T)

    # -- 4. Workgroup sizes --------------------------------------------------
    max_wg = device.limits["max-compute-workgroup-size-x"]
    # Pass 1, 3: one workgroup per patch, threads stripe over D.
    _W = _WARP_SIZE
    W13 = min(((D + _W - 1) // _W) * _W, max_wg)
    W13 = max(W13, _W)
    # Pass 2: Jacobi -- the shader uses a reduce_buf whose size caps the
    # workgroup.
    W2 = min(W13, _MAX_REDUCE_BUF)
    # Pass 4: threads stripe over voxels in the patch.
    W4 = min(((num_samples + _W - 1) // _W) * _W, max_wg)
    W4 = max(W4, _W)

    # -- 5. Persistent output buffers (full volume) --------------------------
    n_voxels = Nx * Ny * Nz
    BU = wgpu.BufferUsage
    STORAGE = BU.STORAGE
    COPY_SRC = BU.COPY_SRC
    COPY_DST = BU.COPY_DST
    UNIFORM = BU.UNIFORM

    gpu_input = device.create_buffer_with_data(
        data=arr_f32.tobytes(),
        usage=STORAGE,
    )
    # output, theta, sigma use atomic u32 in the shader -> sized as u32
    gpu_output = device.create_buffer(
        size=n_voxels * D * 4,
        usage=STORAGE | COPY_SRC,
    )
    gpu_theta = device.create_buffer(
        size=n_voxels * 4,
        usage=STORAGE | COPY_SRC,
    )
    gpu_sigma_out = device.create_buffer(
        size=n_voxels * 4,
        usage=STORAGE | COPY_SRC,
    )
    # Full patch index buffer (read by pass 1 with tile_offset).
    gpu_patch_indices_full = device.create_buffer_with_data(
        data=patch_indices.tobytes(),
        usage=STORAGE | COPY_SRC,
    )

    # -- 6. Per-tile buffers (allocated once, reused) ------------------------
    gpu_cov = device.create_buffer(
        size=T * D * D * 4,
        usage=STORAGE,
    )
    gpu_means = device.create_buffer(
        size=T * D * 4,
        usage=STORAGE,
    )
    gpu_eigenvectors = device.create_buffer(
        size=T * D * D * 4,
        usage=STORAGE,
    )
    gpu_eigenvalues = device.create_buffer(
        size=T * D * 4,
        usage=STORAGE,
    )
    gpu_converged = device.create_buffer(
        size=T * 4,
        usage=STORAGE | COPY_SRC,
    )
    gpu_off_diag = device.create_buffer(
        size=T * 2 * 4,  # two floats per patch (initial, current)
        usage=STORAGE,
    )
    gpu_reconstructed = device.create_buffer(
        size=T * num_samples * D * 4,
        usage=STORAGE,
    )
    gpu_ncomps = device.create_buffer(
        size=T * 4,
        usage=STORAGE,
    )
    gpu_sigma_est = device.create_buffer(
        size=T * 4,
        usage=STORAGE,
    )
    # Tile-local patch indices for pass 4 (accumulate uses patch_idx * 6u
    # without a tile_offset, so it needs a buffer containing only the
    # current tile's indices).
    gpu_patch_indices_tile = device.create_buffer(
        size=T * 6 * 4,
        usage=STORAGE | COPY_DST,
    )

    # -- 7. Uniform buffers (one per pass, rewritten each tile) ---------------
    gpu_uniform_cov = device.create_buffer(
        size=32, usage=UNIFORM | COPY_DST,
    )
    gpu_uniform_jacobi = device.create_buffer(
        size=16, usage=UNIFORM | COPY_DST,
    )
    gpu_uniform_classify = device.create_buffer(
        size=32, usage=UNIFORM | COPY_DST,
    )
    gpu_uniform_accum = device.create_buffer(
        size=32, usage=UNIFORM | COPY_DST,
    )

    # -- 8. Compile shaders --------------------------------------------------
    shader_cov = device.create_shader_module(
        code=_load_shader("extract_covariance.wgsl"),
    )
    if use_global_mem:
        shader_jacobi = device.create_shader_module(
            code=_load_shader("jacobi_eigh_global.wgsl"),
        )
        shader_classify = device.create_shader_module(
            code=_load_shader("classify_reconstruct_global.wgsl"),
        )
    else:
        shader_jacobi = device.create_shader_module(
            code=_load_shader("jacobi_eigh.wgsl", D=D),
        )
        shader_classify = device.create_shader_module(
            code=_load_shader("classify_reconstruct.wgsl", D=D),
        )
    shader_accum = device.create_shader_module(
        code=_load_shader("accumulate.wgsl"),
    )

    # -- 9. Compute pipelines ------------------------------------------------
    pipeline_cov = device.create_compute_pipeline(
        layout="auto",
        compute={
            "module": shader_cov,
            "entry_point": "main",
            "constants": {"workgroup_size": int(W13)},
        },
    )
    jacobi_constants = {"workgroup_size": int(W2)}
    if not use_global_mem:
        jacobi_constants["d_max"] = int(D)
    pipeline_jacobi = device.create_compute_pipeline(
        layout="auto",
        compute={
            "module": shader_jacobi,
            "entry_point": "main",
            "constants": jacobi_constants,
        },
    )
    pipeline_classify = device.create_compute_pipeline(
        layout="auto",
        compute={
            "module": shader_classify,
            "entry_point": "main",
            "constants": {"workgroup_size": int(W13)},
        },
    )
    pipeline_accum = device.create_compute_pipeline(
        layout="auto",
        compute={
            "module": shader_accum,
            "entry_point": "main",
            "constants": {"workgroup_size": int(W4)},
        },
    )

    # -- 10. Bind groups -----------------------------------------------------
    # Pass 1: params, input_volume, patch_indices(full), cov, means
    bg_layout_cov = pipeline_cov.get_bind_group_layout(0)
    bind_group_cov = device.create_bind_group(
        layout=bg_layout_cov,
        entries=[
            {"binding": 0, "resource": {"buffer": gpu_uniform_cov}},
            {"binding": 1, "resource": {"buffer": gpu_input}},
            {"binding": 2, "resource": {"buffer": gpu_patch_indices_full}},
            {"binding": 3, "resource": {"buffer": gpu_cov}},
            {"binding": 4, "resource": {"buffer": gpu_means}},
        ],
    )

    # Pass 2: params, cov, eigenvectors, eigenvalues, converged, off_diag
    bg_layout_jacobi = pipeline_jacobi.get_bind_group_layout(0)
    bind_group_jacobi = device.create_bind_group(
        layout=bg_layout_jacobi,
        entries=[
            {"binding": 0, "resource": {"buffer": gpu_uniform_jacobi}},
            {"binding": 1, "resource": {"buffer": gpu_cov}},
            {"binding": 2, "resource": {"buffer": gpu_eigenvectors}},
            {"binding": 3, "resource": {"buffer": gpu_eigenvalues}},
            {"binding": 4, "resource": {"buffer": gpu_converged}},
            {"binding": 5, "resource": {"buffer": gpu_off_diag}},
        ],
    )

    # Pass 3: params, input_volume, patch_indices(full), eigenvalues,
    #          eigenvectors, means, reconstructed, ncomps, sigma_est
    # Pass 3 uses tile_offset to index into the full patch_indices buffer.
    bg_layout_classify = pipeline_classify.get_bind_group_layout(0)
    bind_group_classify = device.create_bind_group(
        layout=bg_layout_classify,
        entries=[
            {"binding": 0, "resource": {"buffer": gpu_uniform_classify}},
            {"binding": 1, "resource": {"buffer": gpu_input}},
            {"binding": 2, "resource": {"buffer": gpu_patch_indices_full}},
            {"binding": 3, "resource": {"buffer": gpu_eigenvalues}},
            {"binding": 4, "resource": {"buffer": gpu_eigenvectors}},
            {"binding": 5, "resource": {"buffer": gpu_means}},
            {"binding": 6, "resource": {"buffer": gpu_reconstructed}},
            {"binding": 7, "resource": {"buffer": gpu_ncomps}},
            {"binding": 8, "resource": {"buffer": gpu_sigma_est}},
        ],
    )

    # Pass 4: params, patch_indices(tile), reconstructed, ncomps, sigma_est,
    #          output_volume, theta, sigma_out
    bg_layout_accum = pipeline_accum.get_bind_group_layout(0)
    bind_group_accum = device.create_bind_group(
        layout=bg_layout_accum,
        entries=[
            {"binding": 0, "resource": {"buffer": gpu_uniform_accum}},
            {"binding": 1, "resource": {"buffer": gpu_patch_indices_tile}},
            {"binding": 2, "resource": {"buffer": gpu_reconstructed}},
            {"binding": 3, "resource": {"buffer": gpu_ncomps}},
            {"binding": 4, "resource": {"buffer": gpu_sigma_est}},
            {"binding": 5, "resource": {"buffer": gpu_output}},
            {"binding": 6, "resource": {"buffer": gpu_theta}},
            {"binding": 7, "resource": {"buffer": gpu_sigma_out}},
        ],
    )

    # -- 11. Tile loop -------------------------------------------------------
    try:
        from tqdm import tqdm
        tile_iter = tqdm(
            range(0, n_patches, T), desc="GPU MPPCA", unit="tile",
        )
    except ImportError:
        tile_iter = range(0, n_patches, T)

    for tile_start in tile_iter:
        tile_end = min(tile_start + T, n_patches)
        current_T = tile_end - tile_start

        # Upload tile-local patch indices for pass 4 (accumulate).
        tile_indices_bytes = patch_indices[tile_start:tile_end].tobytes()
        # Pad to full T * 6 * 4 if the last tile is smaller.
        if current_T < T:
            tile_indices_bytes += b"\x00" * ((T - current_T) * 6 * 4)
        device.queue.write_buffer(gpu_patch_indices_tile, 0,
                                  tile_indices_bytes)

        device.queue.write_buffer(
            gpu_uniform_cov, 0,
            _pack_volume_params(D, Nx, Ny, Nz, num_samples,
                               tile_start, current_T),
        )

        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline_cov)
        cp.set_bind_group(0, bind_group_cov)
        cp.dispatch_workgroups(current_T)
        cp.end()
        device.queue.submit([enc.finish()])

        for sweep in range(_MAX_JACOBI_SWEEPS):
            device.queue.write_buffer(
                gpu_uniform_jacobi, 0,
                _pack_jacobi_params(D, current_T, sweep, _JACOBI_EPSILON),
            )
            enc = device.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipeline_jacobi)
            cp.set_bind_group(0, bind_group_jacobi)
            cp.dispatch_workgroups(current_T)
            cp.end()
            device.queue.submit([enc.finish()])

        device.queue.write_buffer(
            gpu_uniform_classify, 0,
            _pack_volume_params(D, Nx, Ny, Nz, num_samples,
                               tile_start, current_T),
        )

        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline_classify)
        cp.set_bind_group(0, bind_group_classify)
        cp.dispatch_workgroups(current_T)
        cp.end()
        device.queue.submit([enc.finish()])

        device.queue.write_buffer(
            gpu_uniform_accum, 0,
            _pack_accumulate_params(D, Nx, Ny, Nz, num_samples, current_T),
        )

        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline_accum)
        cp.set_bind_group(0, bind_group_accum)
        cp.dispatch_workgroups(current_T)
        cp.end()
        device.queue.submit([enc.finish()])

    # -- 12. Read back results ------------------------------------------------
    output_bytes = device.queue.read_buffer(gpu_output)
    theta_bytes = device.queue.read_buffer(gpu_theta)

    # The output and theta buffers store bitwise u32 representations of
    # floats accumulated via atomic CAS.  Reinterpret as float32.
    output_u32 = np.frombuffer(output_bytes, dtype=np.uint32)
    output_f32 = output_u32.view(np.float32).reshape(Nx, Ny, Nz, D).copy()

    theta_u32 = np.frombuffer(theta_bytes, dtype=np.uint32)
    theta_f32 = theta_u32.view(np.float32).reshape(Nx, Ny, Nz).copy()

    # Weighted average (avoid division by zero).
    nonzero = theta_f32 > 0.0
    output_f32[nonzero] /= theta_f32[nonzero, np.newaxis]

    # Zero outside mask.
    output_f32[~mask] = 0.0

    # Clip to non-negative values (same as CPU path).
    np.clip(output_f32, 0.0, None, out=output_f32)

    result = output_f32.astype(out_dtype)

    if return_sigma:
        sigma_bytes = device.queue.read_buffer(gpu_sigma_out)
        sigma_u32 = np.frombuffer(sigma_bytes, dtype=np.uint32)
        sigma_f32 = sigma_u32.view(np.float32).reshape(Nx, Ny, Nz).copy()
        # sigma_f32 contains accumulated (variance * weight).  Divide by
        # theta to get the weighted average variance, then take sqrt to
        # match the CPU which returns sqrt(var).
        sigma_f32[nonzero] /= theta_f32[nonzero]
        sigma_f32[~mask] = 0.0
        return result, np.sqrt(sigma_f32).astype(out_dtype)

    return result
