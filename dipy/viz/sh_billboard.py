
from __future__ import annotations

from math import ceil

from fury.actor import Group, Mesh  # noqa: F401
from fury.geometry import buffer_to_geometry
from fury.lib import register_wgpu_render_function
from fury.material import (
    MeshPhongMaterial,
    SphGlyphMaterial,
    validate_opacity,
)
import fury.primitive as fp
from fury.shader import (
    BaseShader,
    Binding,
    Buffer,
    MeshShader,
)
from fury.utils import create_sh_basis_matrix, get_lmax, get_n_coeffs
import numpy as np
import wgpu

from dipy.viz.wgsl import load_dipy_wgsl


def _sh_basis_for_basis_type(directions, l_max, basis_type="standard"):
    """SH basis matrix adapted for *basis_type* (handles descoteaux07)."""
    basis = create_sh_basis_matrix(directions, l_max)
    if basis_type == "descoteaux07":
        even_cols = []
        for big_l in range(0, l_max + 1, 2):
            for m in range(-big_l, big_l + 1):
                even_cols.append(big_l * big_l + big_l + m)
        basis = basis[:, even_cols]
    return basis


_SH_PRECOMPUTE_CACHE: dict = {}
_GPU_DEVICE_LIMITS_CACHE: dict = {}

_MAX_LUT_CHUNKS = 8


def _get_gpu_max_buffer_size() -> int:
    if "max_storage_buffer_binding_size" in _GPU_DEVICE_LIMITS_CACHE:
        return _GPU_DEVICE_LIMITS_CACHE["max_storage_buffer_binding_size"]

    try:
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
        device = adapter.request_device_sync()
        limits = device.limits
        max_size = limits.get(
            "max-storage-buffer-binding-size", 128 * 1024 * 1024
        )
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


class BillboardSphereMaterial(MeshPhongMaterial):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _SphGlyphLutMaterial(SphGlyphMaterial):

    def __init__(self, auto_detach=True, **kwargs):
        super().__init__(**kwargs)
        self.auto_detach = bool(auto_detach)


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
            raise ValueError(
                "The attribute 'l_max' must be a non-negative integer."
            )
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


class BillboardSphereShader(MeshShader):

    def __init__(self, wobject):
        super().__init__(wobject)
        self["billboard_count"] = getattr(wobject, "billboard_count", 1)
        self["lighting"] = "phong"

    def get_code(self):
        return load_dipy_wgsl("billboard_sphere_render.wgsl")


class BillboardSphGlyphLutComputeShader(BaseShader):

    type = "compute"

    def __init__(self, wobject):
        super().__init__(wobject)
        self._wobject = wobject
        self["n_coeffs"] = getattr(wobject, "coeffs_per_glyph", 0)
        self["l_max"] = getattr(wobject, "_l_max", 0)
        self["theta_res"] = getattr(wobject, "_sh_lut_theta_res", 0)
        self["phi_res"] = getattr(wobject, "_sh_lut_phi_res", 0)
        self["glyph_count"] = getattr(wobject, "billboard_count", 0)
        self["workgroup_size"] = 128
        self["entry_point"] = "compute_lut"

    def get_render_info(self, _wobject, _shared):
        if not getattr(self._wobject, "_sh_lut_needs_dispatch", False):
            return {"indices": (0, 0, 0)}

        theta_res = max(int(self["theta_res"]), 0)
        phi_res = max(int(self["phi_res"]), 0)
        glyph_count = max(int(self["glyph_count"]), 0)

        if theta_res <= 0 or phi_res <= 0 or glyph_count <= 0:
            return {"indices": (0, 0, 0)}

        theta_groups = int(ceil(theta_res / 8.0))
        phi_groups = int(ceil(phi_res / 8.0))

        self._wobject._sh_lut_ready = True
        self._wobject._sh_lut_needs_dispatch = False
        self._wobject._sh_use_radius_lut = True
        return {"indices": (glyph_count, theta_groups, phi_groups)}

    def get_pipeline_info(self, _wobject, _shared):
        return {}

    def get_bindings(self, wobject, _shared, _scene=None):
        coeff_buffer = getattr(wobject, "sh_coeffs_buffer", None)
        if coeff_buffer is None:
            coeff_buffer = Buffer(wobject.sh_coeffs)
            wobject.sh_coeffs_buffer = coeff_buffer

        radius_buffer = getattr(wobject, "_sh_radius_lut_buffer", None)

        n_glyphs = getattr(wobject, "billboard_count", 0)
        n_coeffs = getattr(wobject, "coeffs_per_glyph", 0)
        phi_res = getattr(wobject, "_sh_lut_phi_res", 0)
        theta_res = getattr(wobject, "_sh_lut_theta_res", 0)
        l_max = getattr(wobject, "_l_max", 4)

        uniforms_dtype = np.dtype(
            [
                ("n_glyphs", "u4"),
                ("n_coeffs", "u4"),
                ("phi_res", "u4"),
                ("theta_res", "u4"),
                ("l_max", "u4"),
                ("_pad1", "u4"),
                ("_pad2", "u4"),
                ("_pad3", "u4"),
            ]
        )
        uniforms_data = np.array(
            [(n_glyphs, n_coeffs, phi_res, theta_res, l_max, 0, 0, 0)],
            dtype=uniforms_dtype,
        )
        uniforms_buffer = Buffer(uniforms_data)

        bindings: dict = {
            0: {
                0: Binding(
                    "s_sh_coeffs",
                    "buffer/read_only_storage",
                    coeff_buffer,
                    "COMPUTE",
                ),
            }
        }

        if radius_buffer is not None:
            bindings[0][1] = Binding(
                "s_radius_lut",
                "buffer/storage",
                radius_buffer,
                "COMPUTE",
            )

        bindings[0][2] = Binding(
            "uniforms",
            "buffer/uniform",
            uniforms_buffer,
            "COMPUTE",
        )

        self.define_bindings(0, bindings[0])
        return bindings

    def get_code(self):
        return load_dipy_wgsl("sh_analytical_lut_populate.wgsl")


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
        self["use_precomputation"] = int(
            getattr(wobject, "_is_precomputed", False)
        )
        self["use_level_of_detail"] = int(
            getattr(wobject, "_use_level_of_detail", True)
        )
        use_radius_lut = bool(
            getattr(wobject, "_sh_use_radius_lut", False)
        )
        self["use_precomputed_radius_lut"] = (
            "true" if use_radius_lut else "false"
        )

        interp_mode = getattr(wobject, "_sh_interpolation_mode", None)
        if interp_mode is None:
            use_bicubic = bool(
                getattr(wobject, "_sh_use_bicubic", False)
            )
            interp_mode = 2 if use_bicubic else 1
        self["interpolation_mode"] = int(interp_mode)

        self["radius_lut_theta"] = getattr(
            wobject, "_sh_lut_theta_res", 0
        )
        self["radius_lut_phi"] = getattr(wobject, "_sh_lut_phi_res", 0)
        self["radius_lut_stride"] = getattr(wobject, "_sh_lut_stride", 0)
        self["radius_theta_step"] = getattr(
            wobject, "_sh_theta_step", 0.0
        )
        self["radius_phi_step"] = getattr(wobject, "_sh_phi_step", 0.0)
        self["lut_n_chunks"] = getattr(wobject, "_sh_lut_n_chunks", 1)
        self["lut_glyphs_per_chunk"] = getattr(
            wobject, "_sh_lut_glyphs_per_chunk", 0
        )
        self["debug_mode"] = getattr(wobject, "_sh_debug_mode", 0)
        force_direct = bool(
            getattr(wobject, "_sh_force_direct_eval", False)
        )
        self["force_direct_sh_eval"] = (
            "true" if force_direct else "false"
        )
        use_octahedral = bool(
            getattr(wobject, "_sh_use_octahedral_lut", False)
        )
        self["use_octahedral_lut"] = (
            "true" if use_octahedral else "false"
        )
        use_hermite = bool(
            getattr(wobject, "_sh_use_hermite_interp", False)
        )
        self["use_hermite_interp"] = (
            "true" if use_hermite else "false"
        )
        force_fd = bool(
            getattr(wobject, "_sh_force_fd_normals", False)
        )
        self["force_fd_normals"] = "true" if force_fd else "false"
        use_float16 = bool(
            getattr(wobject, "_sh_use_float16", False)
        )
        self["use_float16"] = "true" if use_float16 else "false"

        mapping_mode_str = getattr(
            wobject, "_sh_mapping_mode", "octahedral"
        )
        mapping_mode_map = {
            "octahedral": 0,
            "dual_hemi": 1,
            "dual_paraboloid": 2,
            "latlong": 3,
            "fibonacci": 4,
            "cube": 5,
        }
        self["mapping_mode"] = mapping_mode_map.get(mapping_mode_str, 0)

    def get_render_info(self, wobject, shared):
        try:
            mat = getattr(wobject, "material", None)
            lut_ready = bool(getattr(wobject, "_sh_lut_ready", False))
            auto_detach = bool(getattr(mat, "auto_detach", True))
            if (
                isinstance(mat, _SphGlyphLutMaterial)
                and auto_detach
                and lut_ready
            ):
                baked_mat = SphGlyphMaterial(
                    n_coeffs=int(getattr(mat, "n_coeffs", -1)),
                    scale=float(getattr(mat, "scale", 1.0)),
                    shininess=int(getattr(mat, "shininess", 30)),
                    opacity=float(getattr(mat, "opacity", 1.0)),
                    pick_write=bool(getattr(mat, "pick_write", True)),
                    flat_shading=bool(
                        getattr(mat, "flat_shading", False)
                    ),
                )
                wobject.material = baked_mat
        except Exception:
            pass

        render_info = super().get_render_info(wobject, shared)
        if not render_info or render_info.get("indices") is None:
            geometry = wobject.geometry
            vertex_count = getattr(
                getattr(geometry, "positions", None), "nitems", 0
            )
            if vertex_count <= 0:
                return {"indices": (0, 1, 0, 0)}
            n_instances = 1
            if self.get("instanced"):
                instance_buffer = getattr(
                    wobject, "instance_buffer", None
                )
                n_instances = (
                    getattr(instance_buffer, "nitems", 1) or 1
                )
            render_info = {
                "indices": (vertex_count, int(n_instances), 0, 0)
            }
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
        self.define_bindings(2, coeff_bindings)
        bindings[2] = coeff_bindings

        radius_buffers = getattr(
            wobject, "_sh_radius_lut_buffers", None
        )
        normal_buffer = getattr(wobject, "_sh_normal_lut_buffer", None)

        if normal_buffer is None:
            normal_buffer = Buffer(
                np.zeros((1, 3), dtype=np.float32)
            )

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
            radius_buffer = getattr(
                wobject, "_sh_radius_lut_buffer", None
            )
            if radius_buffer is None:
                radius_buffer = Buffer(
                    np.array([0.0], dtype=np.float32)
                )
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

        hermite_buffers = getattr(
            wobject, "_sh_hermite_lut_buffers", None
        )
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
    actor_cls=None,
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
        np.column_stack(
            [sizes, np.ones((n, 1), dtype=np.float32)]
        ),
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

    actor_type = Billboard if actor_cls is None else actor_cls
    obj = actor_type(geometry=geometry, material=material)
    obj.billboard_count = n
    obj.billboard_centers = centers.copy()
    obj.billboard_sizes = sizes.copy()
    return obj


def _generate_precompute_data(l_max, basis_type="standard"):
    cache_key = f"precompute_{l_max}_{basis_type}"
    if cache_key in _SH_PRECOMPUTE_CACHE:
        return _SH_PRECOMPUTE_CACHE[cache_key]

    if l_max >= 8:
        resolution = (128, 64)
    elif l_max >= 4:
        resolution = (96, 48)
    else:
        resolution = (64, 32)

    precompute_data = {
        "resolution": resolution,
        "l_max": l_max,
        "basis_type": basis_type,
    }

    _SH_PRECOMPUTE_CACHE[cache_key] = precompute_data
    return precompute_data


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
        chunk_data = np.zeros(
            (chunk_glyphs, 6, size, size, 4), dtype=np.float32
        )

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
            chunk_data[:, face, :, :, 0] = values_grid[
                :, start:end, start:end
            ]
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
                du_temp[:, start + 1 : end + 1, :]
                - du_temp[:, start - 1 : end - 1, :]
            ) + c2 * (
                du_temp[:, start + 2 : end + 2, :]
                - du_temp[:, start - 2 : end - 2, :]
            )

        flat_data = chunk_data.reshape(-1, 4)
        if use_float16:
            flat_data = flat_data.astype(np.float16)

        actor._sh_hermite_lut_buffers[chunk_idx].data[:] = flat_data
        actor._sh_hermite_lut_buffers[chunk_idx].update_range()
        glyph_offset += chunk_glyphs

    return True


def _populate_radius_lut_cpu(
    actor, phi_res, theta_res, glyph_count, n_coeffs
):
    theta = np.linspace(0, np.pi, theta_res, dtype=np.float32)
    phi = np.linspace(0, 2 * np.pi, phi_res, dtype=np.float32)
    pp, tt = np.meshgrid(phi, theta)
    x = np.sin(tt) * np.cos(pp)
    y = np.sin(tt) * np.sin(pp)
    z = np.cos(tt)
    dirs = np.column_stack(
        [x.ravel(), y.ravel(), z.ravel()]
    ).astype(np.float32)

    l_max = int(np.sqrt(n_coeffs) - 1)
    basis_matrix = create_sh_basis_matrix(dirs, l_max)
    if basis_matrix.shape[1] > n_coeffs:
        basis_matrix = basis_matrix[:, :n_coeffs]

    coeffs_data = actor.sh_coeffs
    if hasattr(coeffs_data, "data"):
        coeffs_data = coeffs_data.data
    if not isinstance(coeffs_data, np.ndarray):
        coeffs_data = np.asarray(coeffs_data)
    if coeffs_data.ndim == 1:
        coeffs_data = coeffs_data.reshape(-1, n_coeffs)

    radius_lut = actor._sh_radius_lut_buffer.data
    for g in range(glyph_count):
        radii = coeffs_data[g] @ basis_matrix.T
        offset = g * theta_res * phi_res
        radius_lut[offset : offset + theta_res * phi_res] = radii
    actor._sh_radius_lut_buffer.update_full()


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
        wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST
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
            actor._sh_hermite_lut_buffers.append(
                Buffer(hermite_lut, usage=usage)
            )

        actor._sh_hermite_lut_buffer = (
            actor._sh_hermite_lut_buffers[0]
        )
        actor._sh_radius_lut_buffer = None

        if mapping_mode == "cube":
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
            actor._sh_radius_lut_buffers.append(
                Buffer(radius_lut, usage=usage)
            )
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


def _attach_precomputed_radius_tables(actor, precompute_data):
    if (
        not hasattr(actor, "sh_coeffs")
        or getattr(actor, "coeffs_per_glyph", 0) <= 0
    ):
        return

    resolution = precompute_data.get("resolution", (0, 0))
    if len(resolution) != 2:
        return

    n_coeffs = int(actor.coeffs_per_glyph)
    glyph_count = int(getattr(actor, "billboard_count", 0))
    if glyph_count <= 0:
        return

    requested_res = getattr(actor, "_sh_requested_lut_res", None)
    if requested_res is not None and requested_res > 0:
        phi_res = requested_res
        theta_res = requested_res
    else:
        phi_res, theta_res = resolution

    if theta_res <= 0 or phi_res <= 0:
        return

    sample_count = theta_res * phi_res
    chunk_info = _calculate_lut_chunking(glyph_count, sample_count)

    if not chunk_info["feasible"]:
        actor._sh_use_radius_lut = False
        actor._sh_radius_lut_buffer = None
        actor._sh_normal_lut_buffer = None
        actor._sh_lut_ready = True
        return

    n_chunks = chunk_info["n_chunks"]
    usage = (
        wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST
    )

    actor._sh_lut_theta_res = theta_res
    actor._sh_lut_phi_res = phi_res
    actor._sh_lut_stride = sample_count
    actor._sh_theta_step = np.pi / max(theta_res - 1, 1)
    actor._sh_phi_step = (2.0 * np.pi) / max(phi_res, 1)
    actor._sh_lut_n_chunks = n_chunks
    actor._sh_lut_glyphs_per_chunk = chunk_info["glyphs_per_chunk"]
    actor._sh_lut_chunk_sizes = chunk_info["chunk_sizes"]

    actor._sh_radius_lut_buffers = []
    for chunk_glyphs in chunk_info["chunk_sizes"]:
        chunk_samples = chunk_glyphs * sample_count
        radius_lut = np.zeros(chunk_samples, dtype=np.float32)
        actor._sh_radius_lut_buffers.append(
            Buffer(radius_lut, usage=usage)
        )

    actor._sh_radius_lut_buffer = actor._sh_radius_lut_buffers[0]

    dummy_normal = np.zeros((1, 4), dtype=np.float32)
    actor._sh_normal_lut_buffer = Buffer(dummy_normal, usage=usage)

    actor._sh_use_radius_lut = True
    actor._sh_lut_ready = False
    actor._sh_lut_needs_dispatch = True

    old_mat = actor.material
    actor.material = _SphGlyphLutMaterial(
        n_coeffs=int(getattr(old_mat, "n_coeffs", -1)),
        scale=float(getattr(old_mat, "scale", 1.0)),
        shininess=int(getattr(old_mat, "shininess", 30)),
        opacity=float(getattr(old_mat, "opacity", 1.0)),
        auto_detach=True,
    )


def _create_sph_harmonics_billboard(
    coeffs,
    *,
    centers=None,
    sphere=None,
    basis_type="standard",
    color_type="sign",
    l_max=None,
    scale=1.0,
    shininess=50,
    opacity=None,
    enable_picking=True,
    tight_fit=False,
    use_bicubic=False,
    interpolation_mode=None,
    debug_mode=0,
    force_direct_eval=False,
    mapping_mode="latlong",
    lut_res=8,
    use_hermite=False,
    use_float16=False,
):
    coeffs_arr = np.asarray(coeffs, dtype=np.float32)
    if coeffs_arr.ndim == 4:
        n_coeff = coeffs_arr.shape[-1]
    elif coeffs_arr.ndim == 2:
        n_coeff = coeffs_arr.shape[1]
    else:
        raise ValueError("coeffs must be (X,Y,Z,N) or (M,N)")

    inferred_l_max = get_lmax(n_coeff, basis_type=basis_type)
    effective_l_max = l_max if l_max is not None else inferred_l_max

    cache_key = (effective_l_max, basis_type, 128, 64, True)
    if cache_key in _SH_PRECOMPUTE_CACHE:
        precompute_data = _SH_PRECOMPUTE_CACHE[cache_key]
    else:
        precompute_data = _generate_precompute_data(
            effective_l_max, basis_type
        )
        _SH_PRECOMPUTE_CACHE[cache_key] = precompute_data

    if interpolation_mode is None:
        interpolation_mode = 2 if use_bicubic else 1

    actor = sph_glyph_billboard(
        coeffs,
        centers=centers,
        sphere=sphere,
        basis_type=basis_type,
        color_type=color_type,
        l_max=l_max,
        scale=scale,
        shininess=shininess,
        opacity=opacity,
        enable_picking=enable_picking,
        use_precomputation=False,
        tight_fit=tight_fit,
        debug_mode=debug_mode,
        force_direct_eval=force_direct_eval,
    )

    actor._precompute_data = precompute_data
    actor._is_precomputed = True
    actor._is_optimized = True
    actor._cache_key = cache_key
    actor._use_fast_approximation = effective_l_max <= 12
    actor._use_level_of_detail = True
    actor._use_early_discard = True
    actor._sh_interpolation_mode = int(interpolation_mode)
    actor._sh_debug_mode = debug_mode
    actor._sh_force_direct_eval = force_direct_eval
    actor._sh_use_octahedral_lut = False
    actor._sh_use_hermite_interp = False
    actor._sh_force_fd_normals = False
    actor._sh_mapping_mode = "latlong"
    actor._sh_requested_lut_res = lut_res
    actor._sh_use_nn = False
    actor._sh_nn_weights_path = None
    actor._sh_nn_metadata = None

    if mapping_mode != "latlong":
        enable_octahedral_lut(
            actor,
            lut_res=lut_res,
            mapping_mode=mapping_mode,
            use_hermite=use_hermite,
            use_float16=use_float16,
        )
    else:
        _attach_precomputed_radius_tables(actor, precompute_data)

    return actor


def sph_glyph_billboard(
    coeffs,
    *,
    centers=None,
    sphere=None,
    basis_type="standard",
    color_type="sign",
    l_max=None,
    scale=1.0,
    shininess=50,
    opacity=None,
    enable_picking=True,
    use_precomputation=True,
    tight_fit=False,
    use_bicubic=False,
    interpolation_mode=None,
    debug_mode=0,
    force_direct_eval=False,
    mapping_mode="cube",
    lut_res=8,
    use_hermite=False,
    use_float16=False,
):
    if use_precomputation:
        return _create_sph_harmonics_billboard(
            coeffs,
            centers=centers,
            sphere=sphere,
            basis_type=basis_type,
            color_type=color_type,
            l_max=l_max,
            scale=scale,
            shininess=shininess,
            opacity=opacity,
            enable_picking=enable_picking,
            tight_fit=tight_fit,
            use_bicubic=use_bicubic,
            interpolation_mode=interpolation_mode,
            debug_mode=debug_mode,
            force_direct_eval=force_direct_eval,
            mapping_mode=mapping_mode,
            lut_res=lut_res,
            use_hermite=use_hermite,
            use_float16=use_float16,
        )

    coeffs_arr = np.asarray(coeffs, dtype=np.float32)

    if coeffs_arr.ndim == 4:
        coeffs_flat = coeffs_arr.reshape(-1, coeffs_arr.shape[-1])
        if centers is None:
            grid = (
                np.indices(coeffs_arr.shape[:3], dtype=np.float32)
                .reshape(3, -1)
                .T
            )
            centers_arr = grid
        else:
            centers_arr = np.asarray(centers, dtype=np.float32)
    elif coeffs_arr.ndim == 2:
        if centers is None:
            raise ValueError(
                "centers required when coeffs is two-dimensional."
            )
        centers_arr = np.asarray(centers, dtype=np.float32)
        coeffs_flat = coeffs_arr
    else:
        raise ValueError("coeffs must be (X,Y,Z,N) or (M,N).")

    if centers_arr.ndim != 2 or centers_arr.shape[1] != 3:
        raise ValueError("centers must be (M, 3).")
    if centers_arr.shape[0] != coeffs_flat.shape[0]:
        raise ValueError(
            "centers and coeffs must have matching glyph count."
        )

    n_coeff = coeffs_flat.shape[1]
    inferred_l_max = get_lmax(n_coeff, basis_type=basis_type)

    if l_max is None:
        material_n_coeffs = -1
    else:
        if l_max > inferred_l_max:
            raise ValueError("l_max exceeds degree supported by coeffs.")
        material_n_coeffs = get_n_coeffs(l_max, basis_type=basis_type)

    if sphere is None:
        sphere = "symmetric362"

    if isinstance(sphere, str):
        sphere_vertices, _ = fp.prim_sphere(name=sphere)
    elif (
        isinstance(sphere, tuple)
        and len(sphere) == 2
        and all(isinstance(x, int) for x in sphere)
    ):
        sphere_vertices, _ = fp.prim_sphere(
            gen_faces=True, phi=sphere[0], theta=sphere[1]
        )
    elif (
        isinstance(sphere, tuple)
        and len(sphere) == 2
        and isinstance(sphere[0], np.ndarray)
    ):
        sphere_vertices, _ = sphere
    else:
        raise TypeError("sphere must be a name, (phi,theta), or (verts,faces)")

    basis_matrix = _sh_basis_for_basis_type(
        sphere_vertices, inferred_l_max, basis_type=basis_type,
    )
    if basis_matrix.shape[1] != n_coeff:
        raise ValueError("Mismatch between coefficients and SH basis.")

    radii = coeffs_flat @ basis_matrix.T
    max_radius = np.max(np.abs(radii), axis=1)
    max_radius = np.where(
        max_radius > 1e-6,
        max_radius,
        np.full_like(max_radius, 1e-6),
    )
    padding = 1.2
    sizes = (max_radius * scale * 2.0 * padding).astype(np.float32)
    sizes = np.column_stack([sizes, sizes])

    colors = np.ones((coeffs_flat.shape[0], 3), dtype=np.float32)

    material_kwargs = {
        "flat_shading": False,
        "shininess": shininess,
        "n_coeffs": material_n_coeffs,
        "scale": float(scale),
    }

    obj = _create_billboard_actor(
        centers_arr,
        colors,
        sizes,
        opacity,
        enable_picking,
        material_cls=SphGlyphMaterial,
        material_kwargs=material_kwargs,
        actor_cls=SphGlyphBillboard,
    )

    obj.billboard_radii = max_radius * scale
    obj.billboard_mode = "spherical_harmonic"
    obj.n_coeff = n_coeff
    obj.sh_coeffs = coeffs_flat.reshape(-1).astype(np.float32)
    obj.sh_coeffs_buffer = Buffer(obj.sh_coeffs)
    obj.coeffs_per_glyph = n_coeff
    obj.color_type = 0 if color_type == "sign" else 1
    obj._basis_type = basis_type
    obj._l_max = inferred_l_max
    obj._sh_debug_mode = debug_mode
    obj._sh_force_direct_eval = force_direct_eval
    obj._sh_use_octahedral_lut = False
    obj._sh_use_hermite_interp = False
    obj._sh_force_fd_normals = False
    obj.material.n_coeffs = material_n_coeffs
    if coeffs_arr.ndim == 4:
        obj.data_shape = coeffs_arr.shape[:3]

    return obj


def create_large_scale_sh_billboards(
    coeffs,
    centers=None,
    *,
    lut_res=64,
    mapping_mode="cube",
    use_hermite=True,
    **kwargs,
):
    coeffs = np.asarray(coeffs)

    if coeffs.ndim == 4:
        if centers is not None:
            raise ValueError("Cannot provide centres with 4-D coeffs.")
        grid = (
            np.indices(coeffs.shape[:3], dtype=np.float32)
            .reshape(3, -1)
            .T
        )
        centers = grid
        coeffs = coeffs.reshape(-1, coeffs.shape[-1])

    if mapping_mode == "cube":
        padded_res = lut_res + 2
        samples_per_glyph = 6 * padded_res * padded_res
    elif mapping_mode in ("dual_hemi", "dual_paraboloid"):
        samples_per_glyph = 2 * lut_res * lut_res
    else:
        samples_per_glyph = lut_res * lut_res

    bytes_per_sample = 16 if use_hermite else 4
    bytes_per_glyph = samples_per_glyph * bytes_per_sample
    max_buffer_bytes = _get_gpu_max_buffer_size()
    safe_buffer_bytes = int(max_buffer_bytes * 0.8)
    max_glyphs_per_chunk = max(1, safe_buffer_bytes // bytes_per_glyph)
    max_glyphs_per_actor = max_glyphs_per_chunk * _MAX_LUT_CHUNKS

    total_glyphs = len(coeffs)
    actors = []

    if centers is None:
        return [
            sph_glyph_billboard(
                coeffs,
                centers=None,
                lut_res=lut_res,
                mapping_mode=mapping_mode,
                use_hermite=use_hermite,
                use_precomputation=True,
                **kwargs,
            )
        ]

    centers = np.asarray(centers)

    for start in range(0, total_glyphs, max_glyphs_per_actor):
        end = min(start + max_glyphs_per_actor, total_glyphs)
        actor = sph_glyph_billboard(
            coeffs[start:end],
            centers=centers[start:end],
            lut_res=lut_res,
            mapping_mode=mapping_mode,
            use_hermite=use_hermite,
            use_precomputation=True,
            **kwargs,
        )
        actors.append(actor)

    return actors


@register_wgpu_render_function(SphGlyphBillboard, SphGlyphMaterial)
def _register_sph_glyph_render(wobject):
    return (BillboardSphGlyphShader(wobject),)


@register_wgpu_render_function(SphGlyphBillboard, _SphGlyphLutMaterial)
def _register_sph_glyph_lut_render(wobject):
    return (
        BillboardSphGlyphLutComputeShader(wobject),
        BillboardSphGlyphShader(wobject),
    )
