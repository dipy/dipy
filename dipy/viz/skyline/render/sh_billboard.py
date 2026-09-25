"""GPU billboard pipeline for spherical-harmonic ODF glyphs.

Each ODF glyph is a camera-facing quad; the fragment shader ray-marches
it to find where the view ray meets the SH surface r(omega) =
sum(c_lm * Y_lm(omega)).  Because evaluating that sum per pixel is
expensive, :func:`bake_hermite_lut` pre-bakes it into a cube-map Hermite
LUT that the shader samples instead, falling back to direct evaluation
when no LUT is baked.  See individual function/class docstrings for
details (LUT layout and chunking, shader bindings, etc.).
"""

from math import ceil
from typing import ClassVar

import numpy as np

from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.wgsl import load_dipy_wgsl

fury_trip_msg = (
    "Skyline requires Fury version 2.0.0 or higher."
    " Please upgrade Fury by `pip install -U fury --pre` to use Skyline."
)
fury, has_fury_v2, _ = optional_package(
    "fury",
    min_version="2.0.0",
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

    class _FuryBase:
        uniform_type: ClassVar = {}

    SphGlyphMaterial = Mesh = MeshShader = _FuryBase
    Binding = Buffer = buffer_to_geometry = create_sh_basis_matrix = fury
    get_lmax = get_n_coeffs = validate_opacity = fury
    fp = wgpu = fury

    def register_wgpu_render_function(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


_gpu_cache: dict = {}

_MAX_LUT_CHUNKS = 8


def _get_gpu_max_buffer_size():
    """Return cached ``max_storage_buffer_binding_size`` for the default WGPU adapter.

    Returns
    -------
    int
        Device limit in bytes, falling back to 128 MiB if discovery fails.
    """
    if "max_buffer_size" in _gpu_cache:
        return _gpu_cache["max_buffer_size"]

    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync()
        limits = device.limits
        max_size = limits.get("max-storage-buffer-binding-size", 128 * 1024 * 1024)
        _gpu_cache["max_buffer_size"] = max_size
        return max_size
    except Exception:  # noqa: BLE001
        default = 128 * 1024 * 1024
        _gpu_cache["max_buffer_size"] = default
        return default


def _calculate_lut_chunking(
    glyph_count,
    samples_per_glyph,
    *,
    bytes_per_sample=4,
):
    """Plan LUT buffer chunking so each storage buffer stays within GPU limits.

    Parameters
    ----------
    glyph_count : int
        Number of distinct glyphs sharing the LUT layout.
    samples_per_glyph : int
        Scalar LUT entries per glyph for the active mapping mode.
    bytes_per_sample : int, optional
        Width of each LUT texel in bytes.

    Returns
    -------
    dict
        Fields ``n_chunks``, ``glyphs_per_chunk``, ``chunk_sizes``, ``feasible``, etc.
    """
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
    """SH glyph material with world-space slice positions and visibility flags.

    Parameters
    ----------
    active_slice_x : float, optional
        World-space X coordinate of the visible slice plane; a negative
        value disables X-axis slicing.
    active_slice_y : float, optional
        World-space Y coordinate of the visible slice plane; a negative
        value disables Y-axis slicing.
    active_slice_z : float, optional
        World-space Z coordinate of the visible slice plane; a negative
        value disables Z-axis slicing.
    vis_x : int, optional
        Nonzero to enable X-axis slice visibility, zero to hide it.
    vis_y : int, optional
        Nonzero to enable Y-axis slice visibility, zero to hide it.
    vis_z : int, optional
        Nonzero to enable Z-axis slice visibility, zero to hide it.
    **kwargs
        Forwarded to :class:`fury.material.SphGlyphMaterial`.
    """

    uniform_type = dict(  # noqa: RUF012
        SphGlyphMaterial.uniform_type,
        active_slice_x="f4",
        active_slice_y="f4",
        active_slice_z="f4",
        vis_x="i4",
        vis_y="i4",
        vis_z="i4",
    )

    def __init__(
        self,
        *,
        active_slice_x=-1.0,
        active_slice_y=-1.0,
        active_slice_z=-1.0,
        vis_x=1,
        vis_y=1,
        vis_z=1,
        **kwargs,
    ):
        """Initialize the sliced SH glyph material.

        Parameters
        ----------
        active_slice_x : float, optional
            World-space X coordinate of the visible slice plane; a negative
            value disables X-axis slicing.
        active_slice_y : float, optional
            World-space Y coordinate of the visible slice plane; a negative
            value disables Y-axis slicing.
        active_slice_z : float, optional
            World-space Z coordinate of the visible slice plane; a negative
            value disables Z-axis slicing.
        vis_x : int, optional
            Nonzero to enable X-axis slice visibility, zero to hide it.
        vis_y : int, optional
            Nonzero to enable Y-axis slice visibility, zero to hide it.
        vis_z : int, optional
            Nonzero to enable Z-axis slice visibility, zero to hide it.
        **kwargs
            Forwarded to :class:`fury.material.SphGlyphMaterial`.
        """
        super().__init__(**kwargs)
        self.active_slice_x = active_slice_x
        self.active_slice_y = active_slice_y
        self.active_slice_z = active_slice_z
        self.vis_x = vis_x
        self.vis_y = vis_y
        self.vis_z = vis_z


def _make_uniform_property(name, cast):
    """Build a property that reads and writes one GPU uniform buffer field.

    Parameters
    ----------
    name : str
        Field name in the material's ``uniform_buffer``.
    cast : callable
        Applied to values on both read and write (e.g. ``float``, ``int``).

    Returns
    -------
    property
        Descriptor whose getter casts and returns the buffer field, and
        whose setter casts the value, writes it back, and marks the
        uniform buffer fully dirty via ``update_full``.
    """

    def getter(self):
        return cast(self.uniform_buffer.data[name])

    def setter(self, value):
        self.uniform_buffer.data[name] = cast(value)
        self.uniform_buffer.update_full()

    return property(getter, setter)


for _name in ("active_slice_x", "active_slice_y", "active_slice_z"):
    setattr(SlicedSphGlyphMaterial, _name, _make_uniform_property(_name, float))
for _name in ("vis_x", "vis_y", "vis_z"):
    setattr(SlicedSphGlyphMaterial, _name, _make_uniform_property(_name, int))


class Billboard(Mesh):
    """Base mesh class for instanced glyph billboards (Fury ``Mesh`` subclass)."""


class SphGlyphBillboard(Billboard):
    """Multi-glyph SH billboard with LUT baking and per-glyph coefficient buffers."""

    _basis_type = "standard"

    @property
    def l_max(self):
        """Maximum SH order currently shaded.

        Returns
        -------
        int
            Current SH truncation order, or -1 if never set.
        """
        return getattr(self, "_l_max", -1)

    @l_max.setter
    def l_max(self, value):
        """Truncate shading to the given SH order.

        Parameters
        ----------
        value : int
            New SH truncation order; must be a non-negative integer that
            does not exceed the order supported by the current coefficients.

        Raises
        ------
        ValueError
            If ``value`` is not a non-negative integer, or exceeds the SH
            order supported by the number of coefficients on this billboard.
        """
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
    """Pygfx shader: template variables and bindings for the ODF billboard pipeline.

    Reads flags/dimensions off ``wobject`` (the :class:`SphGlyphBillboard`
    actor) at construction time and exposes them as WGSL template variables
    (``{{ n_coeffs }}``, ``{{ use_hermite_lut }}``, etc.) consumed by
    ``sh_billboard.wgsl``.

    Parameters
    ----------
    wobject : SphGlyphBillboard
        Billboard object rendered by this shader.
    """

    def __init__(self, wobject):
        """Initialize the shader from the billboard's current state.

        Parameters
        ----------
        wobject : SphGlyphBillboard
            Billboard object rendered by this shader.
        """
        super().__init__(wobject)
        self._wobject = wobject
        self["billboard_count"] = getattr(wobject, "billboard_count", 1)
        self["lighting"] = "phong"
        self["n_coeffs"] = getattr(wobject, "coeffs_per_glyph", 0)
        self["l_max"] = getattr(wobject, "_l_max", 0)
        self["color_type"] = getattr(wobject, "color_type", 0)
        lut_ready = bool(getattr(wobject, "_sh_lut_ready", False))
        self["use_hermite_lut"] = "true" if lut_ready else "false"
        use_float16 = bool(getattr(wobject, "_sh_use_float16", False))
        self["use_float16"] = "true" if use_float16 else "false"
        self["radius_lut_phi"] = getattr(wobject, "_sh_lut_phi_res", 0)
        self["radius_lut_stride"] = getattr(wobject, "_sh_lut_stride", 0)
        self["lut_n_chunks"] = getattr(wobject, "_sh_lut_n_chunks", 1)
        self["lut_glyphs_per_chunk"] = getattr(wobject, "_sh_lut_glyphs_per_chunk", 0)

        use_slicing = isinstance(
            getattr(wobject, "material", None), SlicedSphGlyphMaterial
        )
        self["use_slicing"] = "true" if use_slicing else "false"

    def get_render_info(self, wobject, shared):
        """Compute the instance/vertex counts pygfx needs to issue the draw call.

        Falls back to computing them from the geometry's vertex buffer
        when the base ``MeshShader`` doesn't already provide indices
        (e.g. before the geometry has been fully wired up).

        Parameters
        ----------
        wobject : SphGlyphBillboard
            Billboard object being rendered.
        shared : fury.lib.Shared
            Pygfx object holding the shared device and pipeline caches.

        Returns
        -------
        dict
            ``{"indices": (vertex_count, instance_count, 0, 0)}``.
        """
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

    def get_bindings(self, wobject, shared, scene=None):  # pep3102: ignore
        """Wire the SH-coefficient and Hermite-LUT storage buffers.

        Group 2 binding 0 is the flat SH coefficient buffer; group 3
        bindings 0-7 are the (up to 8) Hermite LUT chunk buffers, padded
        out with a shared dummy ``vec4<f32>`` buffer when ``wobject``
        has fewer chunks than that (or hasn't baked a LUT at all), since
        WGSL bindings must all be declared even when unused.

        Parameters
        ----------
        wobject : SphGlyphBillboard
            Billboard object being rendered.
        shared : fury.lib.Shared
            Pygfx object holding the shared device and pipeline caches.
        scene : fury.lib.Scene or None, optional
            Scene the billboard belongs to; forwarded to the base
            ``MeshShader`` implementation when it accepts it.

        Returns
        -------
        dict
            Bindings dict with groups 2 and 3 populated, merged onto
            whatever the base ``MeshShader`` already provided.
        """
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

        hermite_buffers = getattr(wobject, "_sh_hermite_lut_buffers", None)
        dummy_vec4 = Buffer(np.zeros((1, 4), dtype=np.float32))

        lut_bindings: dict = {}
        if hermite_buffers is not None and len(hermite_buffers) > 0:
            for i, buf in enumerate(hermite_buffers):
                lut_bindings[i] = Binding(
                    f"s_sh_hermite_lut_{i}",
                    "buffer/read_only_storage",
                    buf,
                    "FRAGMENT",
                )
            for i in range(len(hermite_buffers), 8):
                lut_bindings[i] = Binding(
                    f"s_sh_hermite_lut_{i}",
                    "buffer/read_only_storage",
                    dummy_vec4,
                    "FRAGMENT",
                )
        else:
            for i in range(8):
                lut_bindings[i] = Binding(
                    f"s_sh_hermite_lut_{i}",
                    "buffer/read_only_storage",
                    dummy_vec4,
                    "FRAGMENT",
                )

        self.define_bindings(3, lut_bindings)
        bindings[3] = lut_bindings
        return bindings

    def get_code(self):
        """Return the (still-templated) WGSL source for this shader.

        Returns
        -------
        str
            WGSL source of ``sh_billboard.wgsl``, with template variables
            such as ``{{ n_coeffs }}`` not yet substituted.
        """
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
    """Build a per-glyph 6-vertex-quad ``SphGlyphBillboard`` geometry + material.

    Broadcasts ``colors``/``sizes`` to match ``centers`` when given as a
    single value, repeats each glyph's data across its 6 quad vertices,
    and stores ``billboard_count``/``billboard_centers``/``billboard_sizes``
    on the returned actor for later use (LUT baking, picking, resizing).

    Parameters
    ----------
    centers : ndarray (N, 3) or (3,)
        World-space glyph centers.
    colors : ndarray (N, 3) or (3,)
        Per-glyph RGB color, broadcast to every glyph when a single color.
    sizes : ndarray or scalar
        Per-glyph 2D quad half-extents; a scalar or a length-2/length-N
        array is broadcast to shape ``(N, 2)``.
    opacity : float or None
        Scalar opacity forwarded to :func:`fury.material.validate_opacity`.
    enable_picking : bool
        Whether the material is created with picking writes enabled.
    material_cls : type
        Fury material class used to construct the billboard's material.
    material_kwargs : dict or None, optional
        Extra keyword arguments forwarded to ``material_cls``.

    Returns
    -------
    SphGlyphBillboard
        Billboard actor with geometry, material, and billboard bookkeeping
        attributes (``billboard_count``, ``billboard_centers``,
        ``billboard_sizes``) set.
    """
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


def _populate_hermite_lut_cube_cpu_chunked(
    actor, lut_res, glyph_count, n_coeffs, chunk_info, *, use_float16=False
):
    """CPU (NumPy) fallback for ``bake_hermite_lut`` when GPU compute is unavailable.

    Evaluates the SH basis on a padded per-face grid, takes a 4th-order
    finite-difference of the raw values to get (value, du, dv, d2uv),
    and writes the result into ``actor``'s already-allocated Hermite LUT
    chunk buffers.

    Parameters
    ----------
    actor : SphGlyphBillboard
        Billboard with populated ``sh_coeffs`` and allocated
        ``_sh_hermite_lut_buffers``.
    lut_res : int
        Cube-map resolution per face edge.
    glyph_count : int
        Total number of glyphs baked across all chunks.
    n_coeffs : int
        Number of SH coefficients per glyph.
    chunk_info : dict
        Chunking plan from :func:`_calculate_lut_chunking`.
    use_float16 : bool, optional
        Store the baked LUT values with reduced precision when True.

    Returns
    -------
    bool
        Always True, indicating the bake completed.
    """
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
    actor, lut_res, glyph_count, n_coeffs, chunk_info, *, use_float16=False
):
    """GPU-accelerated cube-mapped Hermite LUT bake (two-pass compute).

    Pass 1 evaluates SH on an internal padded grid (N+6)² per face.
    Pass 2 computes 4th-order finite-difference derivatives and writes
    (value, du, dv, d²uv) into the output hermite LUT buffer.

    Runs imperatively via ``wgpu`` — no pygfx render-function needed.

    Parameters
    ----------
    actor : SphGlyphBillboard
        Billboard with populated ``sh_coeffs`` and allocated
        ``_sh_hermite_lut_buffers``.
    lut_res : int
        Cube-map resolution per face edge.
    glyph_count : int
        Total number of glyphs baked across all chunks.
    n_coeffs : int
        Number of SH coefficients per glyph.
    chunk_info : dict
        Chunking plan from :func:`_calculate_lut_chunking`.
    use_float16 : bool, optional
        Store the baked LUT values with reduced precision when True.

    Returns
    -------
    bool
        Always True, indicating the bake completed.
    """

    N = lut_res
    g_int = 3
    s_int = N + 2 * g_int  # internal padded size per face
    g_out = 1
    s_out = N + 2 * g_out  # output size per face

    l_max = int(getattr(actor, "_l_max", 4))

    # --- cached device + pipelines ----------------------------------------
    cache = _gpu_cache
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
        p1_total_wg = ceil(total_p1 / wg_size)
        p1_x = min(p1_total_wg, 65535)
        p1_y = ceil(p1_total_wg / 65535)

        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(pass1_pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups(p1_x, p1_y)
        cpass.end()

        # --- dispatch pass 2 (finite-difference hermite) ---
        total_p2 = int(chunk_glyphs) * items_per_glyph_p2
        p2_total_wg = ceil(total_p2 / wg_size)
        p2_x = min(p2_total_wg, 65535)
        p2_y = ceil(p2_total_wg / 65535)

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

    return True


def bake_hermite_lut(actor, *, lut_res=8, force_rebake=False, use_float16=False):
    """Bake a cube-mapped Hermite LUT on ``actor`` if GPU memory allows.

    Parameters
    ----------
    actor : SphGlyphBillboard
        Target billboard with populated ``billboard_count`` and coefficients.
    lut_res : int, optional
        Cube-map resolution per face edge.
    force_rebake : bool, optional
        Recompute even when flags indicate the LUT is ready.
    use_float16 : bool, optional
        Store the Hermite LUT with reduced precision when supported.
    """
    if getattr(actor, "_sh_lut_ready", False) and not force_rebake:
        return

    glyph_count = int(getattr(actor, "billboard_count", 0))
    n_coeffs = int(getattr(actor, "coeffs_per_glyph", 0))
    if glyph_count <= 0 or n_coeffs <= 0:
        return

    padded_res = lut_res + 2
    samples_per_glyph = 6 * padded_res * padded_res
    bytes_per_sample = 8 if use_float16 else 16
    chunk_info = _calculate_lut_chunking(
        glyph_count, samples_per_glyph, bytes_per_sample=bytes_per_sample
    )

    if not chunk_info["feasible"]:
        actor._sh_lut_ready = False
        actor._sh_hermite_lut_buffers = None
        actor._sh_hermite_lut_buffer = None
        actor._sh_lut_n_chunks = 1
        actor._sh_lut_glyphs_per_chunk = 0
        actor._sh_lut_phi_res = 0
        actor._sh_lut_stride = 0
        return

    n_chunks = chunk_info["n_chunks"]
    usage = (
        wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    )

    actor._sh_lut_n_chunks = n_chunks
    actor._sh_lut_glyphs_per_chunk = chunk_info["glyphs_per_chunk"]
    actor._sh_lut_chunk_sizes = chunk_info["chunk_sizes"]

    actor._sh_hermite_lut_buffers = []
    dtype = np.float16 if use_float16 else np.float32
    for chunk_glyphs in chunk_info["chunk_sizes"]:
        chunk_samples = chunk_glyphs * samples_per_glyph
        hermite_lut = np.zeros((chunk_samples, 4), dtype=dtype)
        actor._sh_hermite_lut_buffers.append(Buffer(hermite_lut, usage=usage))
    actor._sh_hermite_lut_buffer = actor._sh_hermite_lut_buffers[0]

    try:
        success = _populate_hermite_lut_cube_gpu(
            actor,
            lut_res,
            glyph_count,
            n_coeffs,
            chunk_info,
            use_float16=use_float16,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("GPU LUT bake failed, falling back to CPU: %s", exc)
        success = _populate_hermite_lut_cube_cpu_chunked(
            actor,
            lut_res,
            glyph_count,
            n_coeffs,
            chunk_info,
            use_float16=use_float16,
        )

    actor._sh_use_float16 = use_float16
    actor._sh_lut_phi_res = padded_res
    actor._sh_lut_stride = samples_per_glyph

    actor._sh_lut_ready = bool(success)


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
    color_type : {"orientation", "sign"}, optional
        Encoding forwarded to the material (sign vs orientation hue).
    l_max : int or None, optional
        Explicit truncation order; inferred from ``coeffs`` when None.
    scale : float, optional
        Uniform billboard size multiplier relative to estimated SH radii.
    shininess : float, optional
        Phong exponent for glyph lighting.
    opacity : float or None, optional
        Initial scalar opacity; forwarded to Fury validation when not None.
    enable_picking : bool, optional
        Whether picking handlers are installed on the billboard mesh.
    lut_res : int, optional
        Cube-map LUT resolution per face edge.

    Returns
    -------
    SphGlyphBillboard
        Configured billboard with baked Hermite LUTs.
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
    # The radii below are evaluated from every supplied coefficient, so the basis
    # has to span ``n_coeff``. ``l_max`` only truncates what the material shades.
    basis_matrix = create_sh_basis_matrix(sphere_verts, inferred_l_max)
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
    obj.n_coeff = n_coeff
    obj.sh_coeffs = coeffs.reshape(-1).astype(np.float32)
    obj.sh_coeffs_buffer = Buffer(obj.sh_coeffs)
    obj.coeffs_per_glyph = n_coeff
    obj.color_type = 0 if color_type == "sign" else 1
    obj._basis_type = "standard"
    obj._l_max = inferred_l_max

    obj.material.n_coeffs = material_n_coeffs

    bake_hermite_lut(obj, lut_res=lut_res)

    return obj


@register_wgpu_render_function(SphGlyphBillboard, SlicedSphGlyphMaterial)
def _register_sliced_sph_glyph_render(wobject):
    """Return the shader used for sliced SH billboard rendering.

    Parameters
    ----------
    wobject : SphGlyphBillboard
        Billboard object pygfx is about to render.

    Returns
    -------
    tuple of BillboardSphGlyphShader
        Single-element tuple containing the shader instance to use.
    """

    return (BillboardSphGlyphShader(wobject),)


if not has_fury_v2:
    (
        SlicedSphGlyphMaterial,
        Billboard,
        SphGlyphBillboard,
        BillboardSphGlyphShader,
        sph_glyph_billboard_sliced,
    ) = (fury,) * 5
