from math import ceil

import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.wgsl import load_dipy_wgsl

fury_trip_msg = (
    "Skyline requires Fury version 2.0.0a6 or higher."
    " Please upgrade Fury by `pip install -U fury --pre` to use Skyline."
)
fury, has_fury_v2, _ = optional_package(
    "fury",
    min_version="2.0.0a6",
    trip_msg=fury_trip_msg,
)
if has_fury_v2:
    from fury.actor import Mesh
    from fury.geometry import buffer_to_geometry
    from fury.lib import register_wgpu_render_function
    from fury.material import (
        SphGlyphMaterial,
        validate_opacity,
    )
    import fury.primitive as fp
    from fury.shader import (
        Binding,
        Buffer,
        MeshShader,
    )
    from fury.utils import create_sh_basis_matrix, get_lmax, get_n_coeffs
    import wgpu
else:
    actor = fury.actor

_GPU_DEVICE_LIMITS_CACHE: dict = {}
_GPU_HERMITE_COMPUTE_CACHE: dict = {}

_MAX_LUT_CHUNKS = 8


def _get_gpu_max_buffer_size() -> int:
    if "max_storage_buffer_binding_size" in _GPU_DEVICE_LIMITS_CACHE:
        return _GPU_DEVICE_LIMITS_CACHE["max_storage_buffer_binding_size"]

    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync()
        limits = device.limits
        max_size = limits.get("max-storage-buffer-binding-size", 128 * 1024 * 1024)
        _GPU_DEVICE_LIMITS_CACHE["max_storage_buffer_binding_size"] = max_size
        return max_size
    except Exception:
        default = 128 * 1024 * 1024
        _GPU_DEVICE_LIMITS_CACHE["max_storage_buffer_binding_size"] = default
        return default


def _calculate_lut_chunking(
    glyph_count: int,
    samples_per_glyph: int,
    bytes_per_sample: int = 4,
) -> dict:
    max_buffer_bytes = _get_gpu_max_buffer_size()
    usable_bytes = int(max_buffer_bytes * 0.90)

    total_samples = glyph_count * samples_per_glyph
    total_bytes = total_samples * bytes_per_sample

    if total_bytes <= usable_bytes:
        return {
            "n_chunks": 1,
            "glyphs_per_chunk": glyph_count,
            "samples_per_chunk": total_samples,
            "chunk_sizes": [glyph_count],
            "total_samples": total_samples,
            "feasible": True,
        }

    samples_per_chunk = usable_bytes // bytes_per_sample
    glyphs_per_chunk = samples_per_chunk // samples_per_glyph

    if glyphs_per_chunk < 1:
        return {
            "n_chunks": 0,
            "glyphs_per_chunk": 0,
            "samples_per_chunk": 0,
            "chunk_sizes": [],
            "total_samples": total_samples,
            "feasible": False,
        }

    n_chunks = (glyph_count + glyphs_per_chunk - 1) // glyphs_per_chunk
    chunk_sizes = []
    remaining = glyph_count
    for _ in range(n_chunks):
        cg = min(glyphs_per_chunk, remaining)
        chunk_sizes.append(cg)
        remaining -= cg

    return {
        "n_chunks": n_chunks,
        "glyphs_per_chunk": glyphs_per_chunk,
        "samples_per_chunk": glyphs_per_chunk * samples_per_glyph,
        "chunk_sizes": chunk_sizes,
        "total_samples": total_samples,
        "feasible": n_chunks <= _MAX_LUT_CHUNKS,
    }


class SlicedSphGlyphMaterial(SphGlyphMaterial):
    """SphGlyphMaterial with per-axis slice selection uniforms.

    Uniforms ``active_slice_x/y/z`` select which slice index is visible
    on each axis, and ``vis_x/y/z`` toggle axis visibility.  A glyph is
    rendered when *any* visible axis matches its stored voxel coordinate.
    """

    uniform_type = dict(
        SphGlyphMaterial.uniform_type,
        active_slice_x="i4",
        active_slice_y="i4",
        active_slice_z="i4",
        vis_x="i4",
        vis_y="i4",
        vis_z="i4",
    )

    def __init__(
        self,
        active_slice_x=-1,
        active_slice_y=-1,
        active_slice_z=-1,
        vis_x=1,
        vis_y=1,
        vis_z=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.active_slice_x = active_slice_x
        self.active_slice_y = active_slice_y
        self.active_slice_z = active_slice_z
        self.vis_x = vis_x
        self.vis_y = vis_y
        self.vis_z = vis_z

    def _set_i4(self, name, value):
        self.uniform_buffer.data[name] = int(value)
        self.uniform_buffer.update_full()

    def _get_i4(self, name):
        return int(self.uniform_buffer.data[name])

    @property
    def active_slice_x(self):
        return self._get_i4("active_slice_x")

    @active_slice_x.setter
    def active_slice_x(self, v):
        self._set_i4("active_slice_x", v)

    @property
    def active_slice_y(self):
        return self._get_i4("active_slice_y")

    @active_slice_y.setter
    def active_slice_y(self, v):
        self._set_i4("active_slice_y", v)

    @property
    def active_slice_z(self):
        return self._get_i4("active_slice_z")

    @active_slice_z.setter
    def active_slice_z(self, v):
        self._set_i4("active_slice_z", v)

    @property
    def vis_x(self):
        return self._get_i4("vis_x")

    @vis_x.setter
    def vis_x(self, v):
        self._set_i4("vis_x", v)

    @property
    def vis_y(self):
        return self._get_i4("vis_y")

    @vis_y.setter
    def vis_y(self, v):
        self._set_i4("vis_y", v)

    @property
    def vis_z(self):
        return self._get_i4("vis_z")

    @vis_z.setter
    def vis_z(self, v):
        self._set_i4("vis_z", v)


class Billboard(Mesh):
    pass


class SphGlyphBillboard(Billboard):
    _basis_type = "standard"

    @property
    def l_max(self):
        return getattr(self, "_l_max", -1)

    @l_max.setter
    def l_max(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("The attribute 'l_max' must be a non-negative integer.")
        max_supported = get_lmax(
            getattr(self, "n_coeff", 0),
            basis_type=self._basis_type,
        )
        if value > max_supported:
            raise ValueError(
                "The provided 'l_max' exceeds the number of "
                "spherical harmonic coefficients."
            )
        self._l_max = value
        n_coeffs = get_n_coeffs(value, basis_type=self._basis_type)
        self.material.n_coeffs = n_coeffs


class BillboardSphGlyphShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self._wobject = wobject
        self["billboard_count"] = getattr(wobject, "billboard_count", 1)
        self["lighting"] = "phong"
        original_lmax = getattr(wobject, "_l_max", 0)
        self["n_coeffs"] = getattr(wobject, "coeffs_per_glyph", 0)
        self["l_max"] = original_lmax
        self["color_type"] = getattr(wobject, "color_type", 0)
        self["use_precomputation"] = int(getattr(wobject, "_is_precomputed", False))
        self["use_level_of_detail"] = int(
            getattr(wobject, "_use_level_of_detail", True)
        )
        use_radius_lut = bool(getattr(wobject, "_sh_use_radius_lut", False))
        self["use_precomputed_radius_lut"] = "true" if use_radius_lut else "false"

        interp_mode = getattr(wobject, "_sh_interpolation_mode", None)
        if interp_mode is None:
            use_bicubic = bool(getattr(wobject, "_sh_use_bicubic", False))
            interp_mode = 2 if use_bicubic else 1
        self["interpolation_mode"] = int(interp_mode)

        self["radius_lut_theta"] = getattr(wobject, "_sh_lut_theta_res", 0)
        self["radius_lut_phi"] = getattr(wobject, "_sh_lut_phi_res", 0)
        self["radius_lut_stride"] = getattr(wobject, "_sh_lut_stride", 0)
        self["radius_theta_step"] = getattr(wobject, "_sh_theta_step", 0.0)
        self["radius_phi_step"] = getattr(wobject, "_sh_phi_step", 0.0)
        self["lut_n_chunks"] = getattr(wobject, "_sh_lut_n_chunks", 1)
        self["lut_glyphs_per_chunk"] = getattr(wobject, "_sh_lut_glyphs_per_chunk", 0)
        self["debug_mode"] = getattr(wobject, "_sh_debug_mode", 0)
        force_direct = bool(getattr(wobject, "_sh_force_direct_eval", False))
        self["force_direct_sh_eval"] = "true" if force_direct else "false"
        use_octahedral = bool(getattr(wobject, "_sh_use_octahedral_lut", False))
        self["use_octahedral_lut"] = "true" if use_octahedral else "false"
        use_hermite = bool(getattr(wobject, "_sh_use_hermite_interp", False))
        self["use_hermite_interp"] = "true" if use_hermite else "false"
        force_fd = bool(getattr(wobject, "_sh_force_fd_normals", False))
        self["force_fd_normals"] = "true" if force_fd else "false"
        use_float16 = bool(getattr(wobject, "_sh_use_float16", False))
        self["use_float16"] = "true" if use_float16 else "false"

        mapping_mode_str = getattr(wobject, "_sh_mapping_mode", "octahedral")
        mapping_mode_map = {
            "octahedral": 0,
            "dual_hemi": 1,
            "dual_paraboloid": 2,
            "latlong": 3,
            "fibonacci": 4,
            "cube": 5,
        }
        self["mapping_mode"] = mapping_mode_map.get(mapping_mode_str, 0)

        use_slicing = isinstance(
            getattr(wobject, "material", None), SlicedSphGlyphMaterial
        )
        self["use_slicing"] = "true" if use_slicing else "false"

    def get_render_info(self, wobject, shared):
        render_info = super().get_render_info(wobject, shared)
        if not render_info or render_info.get("indices") is None:
            geometry = wobject.geometry
            vertex_count = getattr(getattr(geometry, "positions", None), "nitems", 0)
            if vertex_count <= 0:
                return {"indices": (0, 1, 0, 0)}
            n_instances = 1
            if self.get("instanced"):
                instance_buffer = getattr(wobject, "instance_buffer", None)
                n_instances = getattr(instance_buffer, "nitems", 1) or 1
            render_info = {"indices": (vertex_count, int(n_instances), 0, 0)}
        return render_info

    def get_bindings(self, wobject, shared, scene=None):
        try:
            bindings = super().get_bindings(wobject, shared, scene)
        except TypeError:
            bindings = super().get_bindings(wobject, shared)

        coeff_buffer = getattr(wobject, "sh_coeffs_buffer", None)
        if coeff_buffer is None:
            coeff_buffer = Buffer(wobject.sh_coeffs)
            wobject.sh_coeffs_buffer = coeff_buffer

        coeff_bindings = {
            0: Binding(
                "s_coeffs",
                "buffer/read_only_storage",
                coeff_buffer,
                "FRAGMENT",
            )
        }

        # Per-glyph slice indices (for sliced rendering)
        slice_buf = getattr(wobject, "slice_indices_buffer", None)
        if slice_buf is not None:
            coeff_bindings[1] = Binding(
                "s_slice_indices",
                "buffer/read_only_storage",
                slice_buf,
                "VERTEX",
            )

        self.define_bindings(2, coeff_bindings)
        bindings[2] = coeff_bindings

        radius_buffers = getattr(wobject, "_sh_radius_lut_buffers", None)
        normal_buffer = getattr(wobject, "_sh_normal_lut_buffer", None)

        if normal_buffer is None:
            normal_buffer = Buffer(np.zeros((1, 3), dtype=np.float32))

        lut_bindings: dict = {}
        dummy_buf = Buffer(np.array([0.0], dtype=np.float32))

        if radius_buffers is not None and len(radius_buffers) > 0:
            for i, buf in enumerate(radius_buffers):
                lut_bindings[i] = Binding(
                    f"s_sh_radius_lut_{i}",
                    "buffer/read_only_storage",
                    buf,
                    "FRAGMENT",
                )
            for i in range(len(radius_buffers), 8):
                lut_bindings[i] = Binding(
                    f"s_sh_radius_lut_{i}",
                    "buffer/read_only_storage",
                    dummy_buf,
                    "FRAGMENT",
                )
        else:
            radius_buffer = getattr(wobject, "_sh_radius_lut_buffer", None)
            if radius_buffer is None:
                radius_buffer = Buffer(np.array([0.0], dtype=np.float32))
            lut_bindings[0] = Binding(
                "s_sh_radius_lut_0",
                "buffer/read_only_storage",
                radius_buffer,
                "FRAGMENT",
            )
            for i in range(1, 8):
                lut_bindings[i] = Binding(
                    f"s_sh_radius_lut_{i}",
                    "buffer/read_only_storage",
                    dummy_buf,
                    "FRAGMENT",
                )

        lut_bindings[8] = Binding(
            "s_sh_normal_lut",
            "buffer/read_only_storage",
            normal_buffer,
            "FRAGMENT",
        )

        hermite_buffers = getattr(wobject, "_sh_hermite_lut_buffers", None)
        dummy_vec4 = Buffer(np.zeros((1, 4), dtype=np.float32))

        if hermite_buffers is not None and len(hermite_buffers) > 0:
            for i, buf in enumerate(hermite_buffers):
                lut_bindings[9 + i] = Binding(
                    f"s_sh_hermite_lut_{i}",
                    "buffer/read_only_storage",
                    buf,
                    "FRAGMENT",
                )
            for i in range(len(hermite_buffers), 8):
                lut_bindings[9 + i] = Binding(
                    f"s_sh_hermite_lut_{i}",
                    "buffer/read_only_storage",
                    dummy_vec4,
                    "FRAGMENT",
                )
        else:
            for i in range(8):
                lut_bindings[9 + i] = Binding(
                    f"s_sh_hermite_lut_{i}",
                    "buffer/read_only_storage",
                    dummy_vec4,
                    "FRAGMENT",
                )

        self.define_bindings(3, lut_bindings)
        bindings[3] = lut_bindings
        return bindings

    def get_code(self):
        return load_dipy_wgsl("sh_billboard.wgsl")


def _create_billboard_actor(
    centers,
    colors,
    sizes,
    opacity,
    enable_picking,
    *,
    material_cls,
    material_kwargs=None,
):
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim == 1:
        centers = centers.reshape(1, 3)
    n = len(centers)

    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim == 1:
        colors = np.tile(colors, (n, 1))
    elif colors.shape[0] != n:
        colors = np.tile(colors[0], (n, 1))

    sizes = np.asarray(sizes, dtype=np.float32)
    if sizes.ndim == 0:
        sizes = np.full((n, 2), float(sizes))
    elif sizes.ndim == 1:
        if sizes.size == 2:
            sizes = np.tile(sizes, (n, 1))
        elif sizes.size == n:
            sizes = np.column_stack([sizes, sizes])
        else:
            sizes = np.full((n, 2), sizes.flat[0])
    elif sizes.shape[0] != n:
        sizes = np.tile(sizes[0], (n, 1))

    opacity = validate_opacity(opacity)

    repeats = 6
    pos = np.repeat(centers, repeats, axis=0).astype(np.float32)
    col = np.repeat(colors, repeats, axis=0).astype(np.float32)
    indices = np.arange(pos.shape[0], dtype=np.uint32).reshape(-1, 3)

    normals = np.repeat(
        np.column_stack([sizes, np.ones((n, 1), dtype=np.float32)]),
        repeats,
        axis=0,
    ).astype(np.float32)

    geometry = buffer_to_geometry(
        positions=pos, colors=col, normals=normals, indices=indices
    )

    material_kwargs = material_kwargs or {}
    material = material_cls(
        pick_write=enable_picking,
        opacity=opacity,
        color_mode="vertex",
        **material_kwargs,
    )

    obj = SphGlyphBillboard(geometry=geometry, material=material)
    obj.billboard_count = n
    obj.billboard_centers = centers.copy()
    obj.billboard_sizes = sizes.copy()
    return obj


def _populate_radius_lut_cube_cpu_chunked(
    actor, lut_res, glyph_count, n_coeffs, chunk_info
):
    padded_res = lut_res + 2
    step = 2.0 / (lut_res - 1)
    u = np.linspace(-1 - step, 1 + step, padded_res, dtype=np.float32)
    v = np.linspace(-1 - step, 1 + step, padded_res, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()

    ones = np.ones_like(uu)
    d0 = np.stack([ones, -vv, -uu], axis=1)
    d1 = np.stack([-ones, -vv, uu], axis=1)
    d2 = np.stack([uu, ones, vv], axis=1)
    d3 = np.stack([uu, -ones, -vv], axis=1)
    d4 = np.stack([uu, -vv, ones], axis=1)
    d5 = np.stack([-uu, -vv, -ones], axis=1)
    dirs = np.concatenate([d0, d1, d2, d3, d4, d5], axis=0)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / norms

    l_max = int(np.sqrt(n_coeffs) - 1)
    basis_matrix = create_sh_basis_matrix(dirs, l_max)
    if basis_matrix.shape[1] > n_coeffs:
        basis_matrix = basis_matrix[:, :n_coeffs]

    glyph_offset = 0
    for chunk_idx, chunk_glyphs in enumerate(chunk_info["chunk_sizes"]):
        radius_lut = actor._sh_radius_lut_buffers[chunk_idx].data
        start_glyph = glyph_offset
        end_glyph = glyph_offset + chunk_glyphs

        coeffs_data = actor.sh_coeffs
        if hasattr(coeffs_data, "data"):
            coeffs_data = coeffs_data.data
        if not isinstance(coeffs_data, np.ndarray):
            coeffs_data = np.asarray(coeffs_data)
        if coeffs_data.ndim == 1:
            if coeffs_data.size % n_coeffs == 0:
                coeffs_data = coeffs_data.reshape(-1, n_coeffs)

        chunk_coeffs = coeffs_data[start_glyph:end_glyph]
        radii = chunk_coeffs @ basis_matrix.T
        radius_lut[:] = radii.flatten()
        actor._sh_radius_lut_buffers[chunk_idx].update_full()
        glyph_offset += chunk_glyphs

    return True


def _populate_hermite_lut_cube_cpu_chunked(
    actor, lut_res, glyph_count, n_coeffs, chunk_info, use_float16=False
):
    N = lut_res
    g = 1
    size = N + 2 * g
    g_internal = 3
    size_internal = N + 2 * g_internal

    step = 2.0 / (N - 1)
    px = np.arange(size_internal, dtype=np.float32)
    py = np.arange(size_internal, dtype=np.float32)
    u_vals = -1.0 + (px - g_internal) * step
    v_vals = -1.0 + (py - g_internal) * step
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu = uu.flatten()
    vv = vv.flatten()

    basis_matrices = []
    for face in range(6):
        if face == 0:
            x, y, z = np.ones_like(uu), -vv, -uu
        elif face == 1:
            x, y, z = -np.ones_like(uu), -vv, uu.copy()
        elif face == 2:
            x, y, z = uu.copy(), np.ones_like(uu), vv.copy()
        elif face == 3:
            x, y, z = uu.copy(), -np.ones_like(uu), -vv
        elif face == 4:
            x, y, z = uu.copy(), -vv, np.ones_like(uu)
        else:
            x, y, z = -uu, -vv, -np.ones_like(uu)

        norm = np.sqrt(x * x + y * y + z * z)
        x /= norm
        y /= norm
        z /= norm
        vertices = np.column_stack((x, y, z))
        basis = create_sh_basis_matrix(vertices, actor._l_max)
        basis_matrices.append(basis)

    glyph_offset = 0
    for chunk_idx, chunk_glyphs in enumerate(chunk_info["chunk_sizes"]):
        coeffs_chunk = actor.sh_coeffs.reshape(-1, n_coeffs)[
            glyph_offset : glyph_offset + chunk_glyphs
        ]
        chunk_data = np.zeros((chunk_glyphs, 6, size, size, 4), dtype=np.float32)

        start = 2
        end = start + size

        c1 = 8.0 / 12.0
        c2 = -1.0 / 12.0

        for face in range(6):
            basis = basis_matrices[face]
            values_face = coeffs_chunk @ basis.T
            values_grid = values_face.reshape(
                chunk_glyphs, size_internal, size_internal
            )
            chunk_data[:, face, :, :, 0] = values_grid[:, start:end, start:end]
            chunk_data[:, face, :, :, 1] = c1 * (
                values_grid[:, start:end, start + 1 : end + 1]
                - values_grid[:, start:end, start - 1 : end - 1]
            ) + c2 * (
                values_grid[:, start:end, start + 2 : end + 2]
                - values_grid[:, start:end, start - 2 : end - 2]
            )
            chunk_data[:, face, :, :, 2] = c1 * (
                values_grid[:, start + 1 : end + 1, start:end]
                - values_grid[:, start - 1 : end - 1, start:end]
            ) + c2 * (
                values_grid[:, start + 2 : end + 2, start:end]
                - values_grid[:, start - 2 : end - 2, start:end]
            )
            du_temp = c1 * (
                values_grid[:, :, start + 1 : end + 1]
                - values_grid[:, :, start - 1 : end - 1]
            ) + c2 * (
                values_grid[:, :, start + 2 : end + 2]
                - values_grid[:, :, start - 2 : end - 2]
            )
            chunk_data[:, face, :, :, 3] = c1 * (
                du_temp[:, start + 1 : end + 1, :] - du_temp[:, start - 1 : end - 1, :]
            ) + c2 * (
                du_temp[:, start + 2 : end + 2, :] - du_temp[:, start - 2 : end - 2, :]
            )

        flat_data = chunk_data.reshape(-1, 4)
        if use_float16:
            flat_data = flat_data.astype(np.float16)

        actor._sh_hermite_lut_buffers[chunk_idx].data[:] = flat_data
        actor._sh_hermite_lut_buffers[chunk_idx].update_range()
        glyph_offset += chunk_glyphs

    return True


def _populate_hermite_lut_cube_gpu(
    actor, lut_res, glyph_count, n_coeffs, chunk_info, use_float16=False
):
    """GPU-accelerated cube-mapped Hermite LUT bake (two-pass compute).

    Pass 1 evaluates SH on an internal padded grid (N+6)² per face.
    Pass 2 computes 4th-order finite-difference derivatives and writes
    (value, du, dv, d²uv) into the output hermite LUT buffer.

    Runs imperatively via ``wgpu`` — no pygfx render-function needed.
    """
    import time as _time

    _t0 = _time.perf_counter()

    N = lut_res
    g_int = 3
    s_int = N + 2 * g_int  # internal padded size per face
    g_out = 1
    s_out = N + 2 * g_out  # output size per face

    l_max = int(getattr(actor, "_l_max", 4))

    # --- cached device + pipelines ----------------------------------------
    cache = _GPU_HERMITE_COMPUTE_CACHE
    if "device" not in cache:
        shader_src = load_dipy_wgsl("sh_cube_hermite_lut_compute.wgsl")
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync(
            required_limits={
                "max-storage-buffer-binding-size": (_get_gpu_max_buffer_size()),
                "max-buffer-size": _get_gpu_max_buffer_size(),
            }
        )
        shader_module = device.create_shader_module(code=shader_src)
        bind_group_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
            ],
        )
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        pass1_pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={
                "module": shader_module,
                "entry_point": "pass1_eval",
            },
        )
        pass2_pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={
                "module": shader_module,
                "entry_point": "pass2_hermite",
            },
        )
        cache["device"] = device
        cache["bind_group_layout"] = bind_group_layout
        cache["pass1_pipeline"] = pass1_pipeline
        cache["pass2_pipeline"] = pass2_pipeline

    device = cache["device"]
    bind_group_layout = cache["bind_group_layout"]
    pass1_pipeline = cache["pass1_pipeline"]
    pass2_pipeline = cache["pass2_pipeline"]

    # --- flatten coefficients for per-chunk upload -------------------------
    coeffs_data = actor.sh_coeffs
    if hasattr(coeffs_data, "data"):
        coeffs_data = coeffs_data.data
    if not isinstance(coeffs_data, np.ndarray):
        coeffs_data = np.asarray(coeffs_data)
    coeffs_flat = coeffs_data.astype(np.float32)

    # --- per-chunk bake ---------------------------------------------------
    glyph_offset = 0
    for chunk_idx, chunk_glyphs in enumerate(chunk_info["chunk_sizes"]):
        # Upload only this chunk's coefficients (avoids alignment issues)
        chunk_coeffs = coeffs_flat[
            glyph_offset * n_coeffs : (glyph_offset + chunk_glyphs) * n_coeffs
        ]
        coeff_chunk_gpu = device.create_buffer_with_data(
            data=chunk_coeffs,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        coeff_chunk_size = int(chunk_glyphs) * int(n_coeffs) * 4
        # Scratch buffer: chunk_glyphs × 6 × s_int × s_int × 4 bytes
        scratch_count = int(chunk_glyphs) * 6 * s_int * s_int
        scratch_gpu = device.create_buffer(
            size=scratch_count * 4,
            usage=wgpu.BufferUsage.STORAGE,
        )

        # Hermite output buffer
        hermite_count = int(chunk_glyphs) * 6 * s_out * s_out
        hermite_byte_size = hermite_count * 16  # vec4<f32>
        hermite_gpu = device.create_buffer(
            size=hermite_byte_size,
            usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC),
        )

        # Uniform buffer
        items_per_glyph_p1 = 6 * s_int * s_int
        items_per_glyph_p2 = 6 * s_out * s_out
        uniforms_dtype = np.dtype(
            [
                ("n_glyphs", "u4"),
                ("n_coeffs", "u4"),
                ("lut_res", "u4"),
                ("l_max", "u4"),
                ("items_per_glyph_p1", "u4"),
                ("items_per_glyph_p2", "u4"),
                ("_pad2", "u4"),
                ("_pad3", "u4"),
            ]
        )
        uniforms_data = np.array(
            [
                (
                    chunk_glyphs,
                    n_coeffs,
                    N,
                    l_max,
                    items_per_glyph_p1,
                    items_per_glyph_p2,
                    0,
                    0,
                )
            ],
            dtype=uniforms_dtype,
        )
        uniform_gpu = device.create_buffer_with_data(
            data=uniforms_data,
            usage=wgpu.BufferUsage.UNIFORM,
        )

        bind_group = device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": coeff_chunk_gpu,
                        "offset": 0,
                        "size": coeff_chunk_size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": hermite_gpu,
                        "offset": 0,
                        "size": hermite_byte_size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": uniform_gpu,
                        "offset": 0,
                        "size": uniforms_data.nbytes,
                    },
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": scratch_gpu,
                        "offset": 0,
                        "size": scratch_count * 4,
                    },
                },
            ],
        )

        # --- dispatch pass 1 (evaluate SH on internal grid) ---
        # 2D dispatch to avoid 65535 limit: total_wg = ceil(items/256)
        # wg_y = ceil(total_wg / 65535), wg_x = min(total_wg, 65535)
        wg_size = 256
        total_p1 = int(chunk_glyphs) * items_per_glyph_p1
        p1_total_wg = int(ceil(total_p1 / wg_size))
        p1_x = min(p1_total_wg, 65535)
        p1_y = int(ceil(p1_total_wg / 65535))

        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(pass1_pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups(p1_x, p1_y)
        cpass.end()

        # --- dispatch pass 2 (finite-difference hermite) ---
        total_p2 = int(chunk_glyphs) * items_per_glyph_p2
        p2_total_wg = int(ceil(total_p2 / wg_size))
        p2_x = min(p2_total_wg, 65535)
        p2_y = int(ceil(p2_total_wg / 65535))

        cpass2 = encoder.begin_compute_pass()
        cpass2.set_pipeline(pass2_pipeline)
        cpass2.set_bind_group(0, bind_group)
        cpass2.dispatch_workgroups(p2_x, p2_y)
        cpass2.end()

        device.queue.submit([encoder.finish()])

        # Read back to CPU via queue.read_buffer (avoids staging buffer)
        raw = device.queue.read_buffer(hermite_gpu)
        hermite_np = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)

        if use_float16:
            hermite_np = hermite_np.astype(np.float16)

        actor._sh_hermite_lut_buffers[chunk_idx].data[:] = hermite_np
        actor._sh_hermite_lut_buffers[chunk_idx].update_range()

        glyph_offset += chunk_glyphs

    _elapsed = _time.perf_counter() - _t0
    return True


def enable_octahedral_lut(
    actor,
    lut_res=64,
    use_hermite=False,
    force_rebake=False,
    mapping_mode="octahedral",
    use_float16=False,
):
    if getattr(actor, "_sh_use_octahedral_lut", False) and not force_rebake:
        return

    glyph_count = int(getattr(actor, "billboard_count", 0))
    n_coeffs = int(getattr(actor, "coeffs_per_glyph", 0))
    if glyph_count <= 0 or n_coeffs <= 0:
        return

    if mapping_mode in ("dual_hemi", "dual_paraboloid"):
        samples_per_glyph = 2 * lut_res * lut_res
    elif mapping_mode == "fibonacci":
        samples_per_glyph = lut_res * lut_res
    elif mapping_mode == "cube":
        padded_res = lut_res + 2
        samples_per_glyph = 6 * padded_res * padded_res
    else:
        samples_per_glyph = lut_res * lut_res

    bytes_per_sample = (8 if use_float16 else 16) if use_hermite else 4
    chunk_info = _calculate_lut_chunking(
        glyph_count, samples_per_glyph, bytes_per_sample
    )

    if not chunk_info["feasible"]:
        actor._sh_use_radius_lut = False
        actor._sh_use_octahedral_lut = False
        actor._sh_lut_ready = True
        return

    n_chunks = chunk_info["n_chunks"]
    usage = (
        wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    )

    actor._sh_lut_n_chunks = n_chunks
    actor._sh_lut_glyphs_per_chunk = chunk_info["glyphs_per_chunk"]
    actor._sh_lut_chunk_sizes = chunk_info["chunk_sizes"]

    success = False
    if use_hermite:
        actor._sh_hermite_lut_buffers = []
        actor._sh_radius_lut_buffers = None
        dtype = np.float16 if use_float16 else np.float32

        for chunk_glyphs in chunk_info["chunk_sizes"]:
            chunk_samples = chunk_glyphs * samples_per_glyph
            hermite_lut = np.zeros((chunk_samples, 4), dtype=dtype)
            actor._sh_hermite_lut_buffers.append(Buffer(hermite_lut, usage=usage))

        actor._sh_hermite_lut_buffer = actor._sh_hermite_lut_buffers[0]
        actor._sh_radius_lut_buffer = None

        if mapping_mode == "cube":
            try:
                success = _populate_hermite_lut_cube_gpu(
                    actor,
                    lut_res,
                    glyph_count,
                    n_coeffs,
                    chunk_info,
                    use_float16=use_float16,
                )
            except Exception:
                success = _populate_hermite_lut_cube_cpu_chunked(
                    actor,
                    lut_res,
                    glyph_count,
                    n_coeffs,
                    chunk_info,
                    use_float16=use_float16,
                )
        else:
            success = False

        actor._sh_use_hermite_interp = True
        actor._sh_use_float16 = use_float16
    else:
        actor._sh_radius_lut_buffers = []
        for chunk_glyphs in chunk_info["chunk_sizes"]:
            chunk_samples = chunk_glyphs * samples_per_glyph
            radius_lut = np.zeros(chunk_samples, dtype=np.float32)
            actor._sh_radius_lut_buffers.append(Buffer(radius_lut, usage=usage))
        actor._sh_radius_lut_buffer = actor._sh_radius_lut_buffers[0]

        if mapping_mode == "cube":
            success = _populate_radius_lut_cube_cpu_chunked(
                actor, lut_res, glyph_count, n_coeffs, chunk_info
            )
        else:
            success = False

        actor._sh_use_hermite_interp = False

    if mapping_mode == "cube":
        actor._sh_lut_theta_res = lut_res + 2
        actor._sh_lut_phi_res = lut_res + 2
    else:
        actor._sh_lut_theta_res = lut_res
        actor._sh_lut_phi_res = lut_res

    actor._sh_mapping_mode = mapping_mode

    if mapping_mode in ("dual_hemi", "dual_paraboloid"):
        actor._sh_lut_stride = 2 * lut_res * lut_res
    elif mapping_mode == "fibonacci":
        actor._sh_lut_stride = lut_res * lut_res
    elif mapping_mode == "cube":
        padded_res = lut_res + 2
        actor._sh_lut_stride = 6 * padded_res * padded_res
    else:
        actor._sh_lut_stride = lut_res * lut_res

    if success:
        actor._sh_use_octahedral_lut = True
        actor._sh_use_radius_lut = True
        actor._sh_lut_ready = True


def sph_glyph_billboard_sliced(
    coeffs,
    centers,
    voxel_coords,
    *,
    color_type="orientation",
    l_max=None,
    scale=1.0,
    shininess=50,
    opacity=None,
    enable_picking=True,
    lut_res=8,
    use_hermite=True,
    mapping_mode="cube",
):
    """Create a *sliced* billboard SH glyph actor.

    Every valid voxel lives in one single actor.  Three uniforms
    (``active_slice_x/y/z``) select which slices are visible;
    switching is a uniform update with zero geometry rebuild.

    A cube-mapped LUT is baked once at creation time so the
    fragment shader uses fast table lookups instead of per-pixel SH
    evaluation.  Chunking is handled automatically by FURY based on
    GPU buffer limits.

    Parameters
    ----------
    coeffs : ndarray (M, n_coeffs)
        Flat SH coefficients for every glyph.
    centers : ndarray (M, 3)
        World-space centres.
    voxel_coords : ndarray (M, 3) int32
        Per-glyph integer voxel (ix, iy, iz).
    lut_res : int
        Cube-map LUT resolution per face edge (default 8).
    use_hermite : bool
        Use Hermite interpolation LUT (default True).
    mapping_mode : str
        LUT mapping mode (default ``"cube"``).
    """
    coeffs = np.asarray(coeffs, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    voxel_coords = np.asarray(voxel_coords, dtype=np.int32)

    n_coeff = coeffs.shape[1]
    inferred_l_max = get_lmax(n_coeff, basis_type="standard")

    if l_max is None:
        material_n_coeffs = -1
    else:
        if l_max > inferred_l_max:
            raise ValueError("l_max exceeds degree supported by coeffs.")
        material_n_coeffs = get_n_coeffs(l_max, basis_type="standard")

    sphere_verts, _ = fp.prim_sphere(name="symmetric362")
    basis_matrix = create_sh_basis_matrix(sphere_verts, l_max)
    if basis_matrix.shape[1] > n_coeff:
        basis_matrix = basis_matrix[:, :n_coeff]

    radii = coeffs @ basis_matrix.T
    max_radius = np.max(np.abs(radii), axis=1)
    max_radius = np.where(max_radius > 1e-6, max_radius, 1e-6)
    padding = 1.2
    sizes = (max_radius * scale * 2.0 * padding).astype(np.float32)
    sizes = np.column_stack([sizes, sizes])

    colors = np.ones((len(coeffs), 3), dtype=np.float32)

    material_kwargs = {
        "flat_shading": False,
        "shininess": shininess,
        "n_coeffs": material_n_coeffs,
        "scale": float(scale),
    }

    obj = _create_billboard_actor(
        centers,
        colors,
        sizes,
        opacity,
        enable_picking,
        material_cls=SlicedSphGlyphMaterial,
        material_kwargs=material_kwargs,
    )

    obj.billboard_radii = max_radius * scale
    obj.billboard_mode = "spherical_harmonic"
    obj.n_coeff = n_coeff
    obj.sh_coeffs = coeffs.reshape(-1).astype(np.float32)
    obj.sh_coeffs_buffer = Buffer(obj.sh_coeffs)
    obj.coeffs_per_glyph = n_coeff
    obj.color_type = 0 if color_type == "sign" else 1
    obj._basis_type = "standard"
    obj._l_max = inferred_l_max
    obj._sh_debug_mode = 0
    obj._sh_force_direct_eval = False
    obj._sh_use_octahedral_lut = False
    obj._sh_use_hermite_interp = False
    obj._sh_force_fd_normals = False
    obj._is_precomputed = True
    obj._is_optimized = True
    obj._use_level_of_detail = True
    obj._use_early_discard = True
    obj._sh_interpolation_mode = 0
    obj._sh_mapping_mode = mapping_mode
    obj._sh_requested_lut_res = lut_res

    obj.slice_indices_buffer = Buffer(voxel_coords.ravel().astype(np.int32))
    obj.material.n_coeffs = material_n_coeffs

    enable_octahedral_lut(
        obj,
        lut_res=lut_res,
        use_hermite=use_hermite,
        mapping_mode=mapping_mode,
    )

    return obj


@register_wgpu_render_function(SphGlyphBillboard, SlicedSphGlyphMaterial)
def _register_sliced_sph_glyph_render(wobject):
    return (BillboardSphGlyphShader(wobject),)
