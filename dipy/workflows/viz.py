from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from dipy.data import read_mni_template
from dipy.io.image import load_nifti
from dipy.io.streamline import load_tractogram
from dipy.io.surface import load_gifti
from dipy.io.utils import split_filename_extension
from dipy.stats.analysis import SpectraGrid
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from dipy.workflows.workflow import Workflow

matplotlib, have_matplotlib, _ = optional_package("matplotlib")
plt, have_plt, _ = optional_package("matplotlib.pyplot")

fury_pckg_msg = (
    "Skyline requires FURY version 2.0.0 or higher. "
    "Please install or upgrade FURY using pip install -U fury --pre."
)
fury, has_fury, _ = optional_package(
    "fury", trip_msg=fury_pckg_msg, min_version="2.0.0"
)
if has_fury:
    from dipy.viz import actor, skyline_from_files, window
else:
    skyline_from_files = fury


class SkylineFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "skyline"

    def run(
        self,
        input_files,
        *,
        rois=None,
        peaks=None,
        odfs=None,
        sh_basis="descoteaux07",
        cluster=False,
        performance_version=False,
        glass_brain=False,
        bg_color=None,
        tract_colors=None,
        cluster_thr=15.0,
        cluster_size_thr=-1,
        cluster_length_thr=-1,
        buan_pvals=None,
        stealth=False,
        rgb=None,
        out_dir="",
        out_stealth_png="out_skyline.png",
    ):
        """Launch Skyline GUI.

        If you want to load only odfs or rois.
        use dipy_skyline run --odfs <Your ODF files> or
        dipy_skyline run --rois <Your ROI files> respectively. You can also load
        peaks with dipy_skyline run --peaks <Your peak files>.

        These options should be used in stealth mode. For GUI mode, the files can be
        loaded through the file dialog.


        Parameters
        ----------
        input_files : variable string or Path
            Tuple of path for each image, peak, surface or tractogram to be added to
            the Skyline viewer.
        rois : variable str, optional
            Tuple of path for each ROI to be added to the Skyline viewer.
        peaks : variable str, optional
            Tuple of path for each peaks file (.pam5, or NIfTI with shape
            (X, Y, Z, 3*N) or (X, Y, Z, N, 3)) to be added to the Skyline viewer.
        odfs : variable str, optional
            Tuple of path for each ODF file (.pam5, or 4D NIfTI of SH
            coefficients) to be added to the Skyline viewer.
        sh_basis : str, optional
            SH basis of NIfTI ODFs: 'descoteaux07' (DIPY legacy) or 'tournier07'
            (MRtrix3).
        cluster : bool, optional
            Whether to cluster the tractograms.
        performance_version : bool, optional
            Whether to use the performance version of the tractogram rendering.
            This will render tractograms as lines instead of tubes,
            which can improve performance for large tractograms.
        glass_brain : bool, optional
            Whether to use glass brain mode. This will overwrite the background color
            to white if not explicitly set by the user.
        bg_color : variable float, optional
            Define the background color of the scene. Colors can be defined with
            3 values and should be between [0-1].
            For example, a value of (0, 0, 0) would mean the black color.
        tract_colors : str, optional
            Define the colors of the tractograms. Colors can be defined with
            3 values and should be between [0-1].
            String options are 'random' for random colors for each tractogram,
            'direction'  for directionally colored streamlines.
            For example, a value of (1, 0, 0) would mean the red color.
        cluster_thr : float, optional
            Distance threshold used for clustering. Default value 15.0 for
            small animal brains you may need to use something smaller such
            as 2.0. The distance is in mm. For this parameter to be active
            ``cluster`` should be enabled.
        cluster_size_thr : int, optional
            Clusters with size less than ``cluster_size_thr`` will be hidden.
            If -1, it will show all cluster above the 50th percentile of the cluster
            size distribution.
        cluster_length_thr : float, optional
            Clusters with average length less than ``cluster_length_thr`` in mm will be
            hidden. If -1, it will show all cluster above the 25th percentile of the
            cluster length distribution.
        buan_pvals : variable str, optional
            File path for BUAN p-values to be used for BUAN-based coloring of
            tractograms.
        stealth : bool, optional
            Capture a figure instead of opening an interactive window.
            ``DIPY_OFFSCREEN=1`` or ``true`` (case-insensitive) also captures when False.
        rgb : bool or None, optional
            ``None``: auto-detect from structured NIfTI ``DT_RGB24`` dtype;
            ``--rgb`` forces RGB; ``--no-rgb`` disables it entirely.
        out_dir : str or Path, optional
            Output directory for offscreen figures.
        out_stealth_png : str, optional
            Filename of the offscreen figure.
        """
        super().__init__(force=True)
        skyline_input_files = []
        start_gui = input_files is not None and input_files[0] in (
            "run",
            "start",
            "launch",
            "initialize",
        )
        if not start_gui:
            io_it = self.get_io_iterator()
            skyline_input_files.extend(input_output[0] for input_output in io_it)

        if cluster_length_thr == -1:
            # set default as None is not allowed in int
            # Further down it will be set to
            # 25th percentile of the cluster length distribution
            cluster_length_thr = None
        if cluster_size_thr == -1:
            # set default as None is not allowed in int
            # Further down it will be set to
            # 50th percentile of the cluster size distribution
            cluster_size_thr = None

        skyline_from_files(
            skyline_input_files,
            rois=rois,
            peaks=peaks,
            shm_coeffs=odfs,
            sh_basis=sh_basis,
            is_cluster=cluster,
            is_light_version=not performance_version,
            glass_brain=glass_brain,
            bg_color=bg_color,
            tract_colors=tract_colors,
            cluster_thr=cluster_thr,
            cluster_size_thr=cluster_size_thr,
            cluster_length_thr=cluster_length_thr,
            buan_pvals=buan_pvals,
            stealth=stealth,
            rgb=rgb,
            out_dir=out_dir,
            out_stealth_png=out_stealth_png,
        )


class HorizonFlow(SkylineFlow):
    @classmethod
    def get_short_name(cls):
        return "horizon"

    def run(
        self,
        input_files,
        cluster=False,
        rgb=None,
        cluster_thr=15.0,
        random_colors=None,
        length_gt=0,
        length_lt=1000,
        clusters_gt=0,
        clusters_lt=10**8,
        native_coords=False,
        stealth=False,
        emergency_header="icbm_2009a",
        bg_color=(0, 0, 0),
        disable_order_transparency=False,
        buan=False,
        buan_thr=0.5,
        buan_highlight=(1, 0, 0),
        roi_images=False,
        roi_colors=(1, 0, 0),
        out_dir="",
        out_stealth_png="tmp.png",
    ):
        """Horizon is deprecated and will be removed with future releases.
        Please use Skyline with `dipy_skyline` CLI.

        See :footcite:p:`Garyfallidis2019` for further details about Horizon.

        Interact with any number of .trx, .trk, .tck or .dpy tractograms and anatomy
        files .nii or .nii.gz. Cluster streamlines on loading.

        Parameters
        ----------
        input_files : variable string or Path
            Filenames.
        cluster : bool, optional
            Enable QuickBundlesX clustering.
        rgb : bool or None, optional
            ``None``: auto-detect from structured NIfTI ``DT_RGB24`` dtype;
            ``--rgb`` forces RGB; ``--no-rgb`` disables it entirely.
        cluster_thr : float, optional
            Distance threshold used for clustering. Default value 15.0 for
            small animal brains you may need to use something smaller such
            as 2.0. The distance is in mm. For this parameter to be active
            ``cluster`` should be enabled.
        random_colors : variable str, optional
            Given multiple tractograms and/or ROIs then each tractogram and/or
            ROI will be shown with different color. If no value is provided,
            both the tractograms and the ROIs will have a different random
            color generated from a distinguishable colormap. If the effect
            should only be applied to one of the 2 types, then use the
            options 'tracts' and 'rois' for the tractograms and the ROIs
            respectively.
            This will be ignored with Skyline.
        length_gt : float, optional
            Clusters with average length greater than ``length_gt`` amount
            in mm will be shown.
        length_lt : float, optional
            Clusters with average length less than ``length_lt`` amount in
            mm will be shown.
            This will be ignored with Skyline.
        clusters_gt : int, optional
            Clusters with size greater than ``clusters_gt`` will be shown.
        clusters_lt : int, optional
            Clusters with size less than ``clusters_lt`` will be shown.
            This will be ignored with Skyline.
        native_coords : bool, optional
            Show results in native coordinates.
            This will be ignored with Skyline.
        stealth : bool, optional
            Do not use interactive mode just save figure.
        emergency_header : str, optional
            If no anatomy reference is provided an emergency header is
            provided. Current options 'icbm_2009a' and 'icbm_2009c'.
            This will be ignored with Skyline.
        bg_color : variable float, optional
            Define the background color of the scene. Colors can be defined
            with 1 or 3 values and should be between [0-1].
        disable_order_transparency : bool, optional
            Use depth peeling to sort transparent objects.
            If True also enables anti-aliasing.
            This will be ignored with Skyline.
        buan : bool, optional
            Enables BUAN framework visualization.
        buan_thr : float, optional
            Uses the threshold value to highlight segments on the
            bundle which have pvalues less than this threshold.
            This will be ignored with Skyline.
        buan_highlight : variable float, optional
            Define the bundle highlight area color. Colors can be defined
            with 1 or 3 values and should be between [0-1].
            For example, a value of (1, 0, 0) would mean the red color.
            This will be ignored with Skyline.
        roi_images : bool, optional
            Displays binary images as contours.
            This will be ignored with Skyline.
        roi_colors : variable float, optional
            Define the color for the roi images. Colors can be defined
            with 1 or 3 values and should be between [0-1]. For example, a
            value of (1, 0, 0) would mean the red color.
            This will be ignored with Skyline.
        out_dir : str or Path, optional
            Output directory.
        out_stealth_png : str, optional
            Filename of saved picture.

        References
        ----------
        .. footbibliography::
        """
        super().__init__(force=True)

        logger.info(
            "Horizon is deprecated and will be removed with future releases. "
            "Please use Skyline with `dipy_skyline` CLI."
        )

        buan_pvals = None
        if buan:
            for input_file in input_files:
                _, ext = split_filename_extension(input_file)
                if ext == ".npy":
                    buan_pvals = [input_file]
                    break

        super().run(
            input_files,
            bg_color=bg_color,
            cluster=cluster,
            rgb=rgb,
            cluster_thr=cluster_thr,
            cluster_length_thr=length_gt,
            cluster_size_thr=clusters_gt,
            buan_pvals=buan_pvals,
            stealth=stealth,
            out_dir=out_dir,
            out_stealth_png=out_stealth_png,
        )


def _grid_values_to_points(values, grid_shape, s_index, r_index):
    """Values of a SPECTRA grid, shape (ns,) or (ns, nr), at each point."""
    values = np.asarray(values, dtype=float).squeeze()
    n_segments, n_radial = grid_shape
    if values.shape not in ((n_segments,), (n_segments, n_radial)):
        raise ValueError(
            f"Grid values of shape {values.shape} do not match the grid of "
            f"{n_segments} segments and {n_radial} radial bins"
        )
    if values.ndim == 1:
        return values[s_index]
    return values[s_index, r_index]


def _values_to_colors(
    values,
    valid,
    *,
    cmap="YlOrBr_r",
    vmin=None,
    vmax=None,
    vcenter=None,
    colorbar_file=None,
    colorbar_title="values",
):
    """Colors of point values with a matplotlib colormap, gray if invalid."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    colors = np.full((len(values), 3), 0.5)
    if not finite.any():
        return colors

    # As in SPECTRA, the default range is the 5th to 95th percentile of the
    # values of all the points
    if vmin is None:
        vmin = np.percentile(values[finite], 5)
    if vmax is None:
        vmax = np.percentile(values[finite], 95)
    if vcenter is not None and vmin < vcenter < vmax:
        norm = matplotlib.colors.TwoSlopeNorm(vcenter, vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    shown = finite & valid
    colors[shown] = mapper.to_rgba(values[shown])[:, :3]

    if colorbar_file is not None:
        fig, ax = plt.subplots(figsize=(8, 1))
        fig.subplots_adjust(bottom=0.5)
        colorbar = fig.colorbar(mapper, cax=ax, orientation="horizontal")
        colorbar.set_label(colorbar_title, size=20)
        ax.tick_params(labelsize=18)
        fig.savefig(colorbar_file, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Colorbar saved as {colorbar_file}")
    return colors


def _alternating_colors(s_index, r_index, valid):
    """Alternating colors of the SPECTRA cells (r_index None along the tract)."""
    tab20 = matplotlib.colormaps["tab20"].colors
    if r_index is None:
        return np.array([tab20[s % len(tab20)] for s in s_index])
    pattern = [tab20[0:2], tab20[2:4]]
    colors = np.array([pattern[s % 2][r % 2] for s, r in zip(s_index, r_index)])
    colors[~valid] = matplotlib.colors.to_rgb("gray")
    return colors


def _direction_colors(lines):
    """Colors of the points from the local direction of their streamline."""
    colors = []
    for line in lines:
        direction = np.abs(np.gradient(np.asarray(line, dtype=float), axis=0))
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        colors.append(direction / np.where(norm > 0, norm, 1))
    return np.concatenate(colors)


def _autocrop(fname, *, border=20):
    """Crop the white border of an image, as in SPECTRA."""
    image = plt.imread(fname)
    content = np.any(image[..., :3] < 1.0, axis=-1)
    if not content.any():
        return
    rows, cols = np.nonzero(content)
    top, left = max(rows.min() - border, 0), max(cols.min() - border, 0)
    bottom = min(rows.max() + border, image.shape[0] - 1)
    right = min(cols.max() + border, image.shape[1] - 1)
    plt.imsave(fname, image[top : bottom + 1, left : right + 1])


def _contour(data, affine, color, opacity):
    """Contour of a mask, shown with the given opacity."""
    contour = actor.contour_from_roi(data, affine=affine, color=color, opacity=opacity)
    # FURY also puts the opacity in the color of the contours, which would
    # apply it twice
    for surface in contour.children:
        surface.material.color = (*color, 1)
    return contour


class SpectraVizFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "spectra_viz"

    def run(
        self,
        bundle_files,
        *,
        ref_bundle=None,
        bundle_color=None,
        color_mask=None,
        segment_length=5.0,
        radial_length=5.0,
        n_segments=None,
        n_radial=None,
        robust_method="linear",
        cmap="YlOrBr_r",
        cmap_title="values",
        vmin=None,
        vmax=None,
        vcenter=None,
        glass_brain=None,
        masks=None,
        mask_colors=None,
        mask_opacity=0.3,
        cam_settings=("Axial",),
        centroid_only=False,
        centroid_with_bundle=False,
        as_points=False,
        linewidth=4.0,
        opacity=1.0,
        no_tube=False,
        auto_crop=False,
        show_bundle=False,
        out_dir="",
        out_image="spectra_bundle.png",
        out_cbar=None,
    ):
        """Workflow of SPECTRA bundle visualization.

        Renders bundles colored by a SPECTRA grid: grid values such as a
        profile or the statistics of each cell, the values of a volume at
        each point, or alternating colors of the grid cells. The grid is
        built from ``ref_bundle`` (e.g. the model bundle), or from each
        bundle if it is not given (see
        :class:`dipy.stats.analysis.SpectraGrid`). A snapshot is saved for
        each camera view.

        See :footcite:p:`Feng2026SPECTRA` for further details about the method.

        Parameters
        ----------
        bundle_files : variable string
            Bundle files to render.
        ref_bundle : string, optional
            Reference (atlas) bundle of the grid. The points of the bundles in
            sparse grid cells are then shown in gray.
        bundle_color : variable string, optional
            Colors of the bundles: one matplotlib color name or one grid
            values file (.npy or .txt with shape (ns,) or (ns, nr)) per
            bundle, a volume (.nii or .nii.gz) sampled at the points, or
            "alternate_1d" or "alternate_2d" for the grid cells. By default
            the bundles are colored by their direction.
        color_mask : string, optional
            Grid mask (.npy with shape (ns,) or (ns, nr)). The cells that are
            False are shown in gray, e.g. the cells that were not tested.
        segment_length : float, optional
            Desired along-tract segment length in millimeters.
        radial_length : float, optional
            Desired radial bin width in millimeters.
        n_segments : int, optional
            Number of along-tract segments, instead of ``segment_length``.
        n_radial : int, optional
            Number of radial bins, instead of ``radial_length``.
        robust_method : string, optional
            Extension of the centroid to the bundle endpoints: "linear",
            "spline", or "none" to use the QuickBundles centroid.
        cmap : string, optional
            Matplotlib colormap of the values.
        cmap_title : string, optional
            Title of the colorbar.
        vmin : float, optional
            Lowest value of the colormap. By default the 5th percentile.
        vmax : float, optional
            Highest value of the colormap. By default the 95th percentile.
        vcenter : float, optional
            Center value of a diverging colormap.
        glass_brain : variable string, optional
            Brain shown in transparency: NIfTI masks, GIfTI surfaces, or
            "default_glass" for the MNI brain mask of DIPY.
        masks : variable string, optional
            Region masks (NIfTI) shown as contours.
        mask_colors : variable string, optional
            Matplotlib color name of each region mask.
        mask_opacity : float, optional
            Opacity of the region masks.
        cam_settings : variable string, optional
            Camera views: Axial, Sagittal_L, Sagittal_R, Coronal_A, Coronal_P.
        centroid_only : bool, optional
            Show only the centroid of the grid, colored like the bundle.
        centroid_with_bundle : bool, optional
            Show the colored centroid over the bundle in light gray.
        as_points : bool, optional
            Show the points of the bundles instead of lines.
        linewidth : float, optional
            Thickness of the streamlines.
        opacity : float, optional
            Opacity of the bundles.
        no_tube : bool, optional
            Show thin lines instead of tubes.
        auto_crop : bool, optional
            Crop the white border of the snapshots.
        show_bundle : bool, optional
            Show the scene in an interactive window.
        out_dir : string or Path, optional
            Output directory.
        out_image : string, optional
            Name of the snapshot. With several cameras, ``_cam01``,
            ``_cam02``, ... is added to the name.
        out_cbar : string, optional
            Name of the colorbar image of the values.

        References
        ----------
        .. footbibliography::

        """
        if not has_fury:
            raise ImportError("dipy_spectra_viz requires FURY 2.0.0 or higher")
        if centroid_only and centroid_with_bundle:
            raise ValueError("Use centroid_only or centroid_with_bundle, not both")
        # Direction and up vector of each camera, as in SPECTRA
        cameras = {
            "Axial": ((0, 0, -1), (0, 1, 0)),
            "Sagittal_L": ((1, 0, 0), (0, 0, 1)),
            "Sagittal_R": ((-1, 0, 0), (0, 0, 1)),
            "Coronal_A": ((0, -1, 0), (0, 0, 1)),
            "Coronal_P": ((0, 1, 0), (0, 0, 1)),
        }
        unknown = [c for c in cam_settings if c not in cameras]
        if unknown:
            raise ValueError(f"Unknown camera(s) {unknown}, use {list(cameras)}")

        bundle_color = list(bundle_color or [])
        spec = bundle_color[0].lower() if bundle_color else ""
        is_values = spec.endswith((".npy", ".txt"))
        is_volume = spec.endswith((".nii", ".nii.gz"))
        is_alternate = spec in ("alternate_1d", "alternate_2d")
        needs_grid = is_values or is_alternate or centroid_only or centroid_with_bundle

        grid_options = {
            "segment_length": segment_length,
            "radial_length": radial_length,
            "n_segments": n_segments,
            "n_radial": n_radial,
            "use_robust_centroid": robust_method != "none",
            "robust_method": robust_method,
        }
        ref_grid = None
        if ref_bundle is not None:
            ref_lines = load_tractogram(
                ref_bundle, reference="same", bbox_valid_check=False
            ).streamlines
            ref_grid = SpectraGrid(ref_lines, **grid_options)
        grid_mask = None if color_mask is None else np.load(color_mask)
        colormap_options = {
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "vcenter": vcenter,
            "colorbar_file": None if out_cbar is None else Path(out_dir) / out_cbar,
            "colorbar_title": cmap_title,
        }
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        scene = window.Scene(background=(1, 1, 1))
        for i, bundle_file in enumerate(bundle_files):
            lines = load_tractogram(
                bundle_file, reference="same", bbox_valid_check=False
            ).streamlines
            logger.info(f"Loaded {len(lines)} streamlines from {bundle_file}")

            # Position of each point in the grid
            if needs_grid:
                grid = ref_grid or SpectraGrid(lines, **grid_options)
                s_index, r_index, _, valid, _ = grid.assign(lines)
                if ref_grid is None:
                    valid = np.ones(len(s_index), dtype=bool)
                if centroid_only or centroid_with_bundle:
                    lines = Streamlines([grid.centroid])
                    s_index = np.arange(grid.n_segments)
                    r_index = np.zeros(grid.n_segments, dtype=int)
                    valid = np.ones(grid.n_segments, dtype=bool)
                grid_shape = (grid.n_segments, grid.n_radial)
                if grid_mask is not None:
                    valid = (
                        _grid_values_to_points(grid_mask, grid_shape, s_index, r_index)
                        > 0
                    )

            # Colors of the points
            if is_values:
                values = (
                    np.load(bundle_color[i])
                    if bundle_color[i].endswith(".npy")
                    else np.loadtxt(bundle_color[i])
                )
                point_values = _grid_values_to_points(
                    values, grid_shape, s_index, r_index
                )
                colors = _values_to_colors(point_values, valid, **colormap_options)
            elif is_volume:
                volume, affine = load_nifti(bundle_color[0])
                points = transform_streamlines(lines, np.linalg.inv(affine))
                point_values = map_coordinates(
                    volume, Streamlines(points).get_data().T, order=1
                )
                colors = _values_to_colors(
                    point_values,
                    np.ones(len(point_values), dtype=bool),
                    **colormap_options,
                )
            elif is_alternate:
                along_only = spec == "alternate_1d"
                colors = _alternating_colors(
                    s_index, None if along_only else r_index, valid
                )
            elif bundle_color:
                colors = matplotlib.colors.to_rgb(bundle_color[i])
            else:
                colors = _direction_colors(lines)

            if centroid_with_bundle:
                bundle_lines = load_tractogram(
                    bundle_file, reference="same", bbox_valid_check=False
                ).streamlines
                background = actor.streamlines(
                    bundle_lines, colors=(0.83, 0.83, 0.83), opacity=0.1
                )
                # The transparent bundle must not hide the centroid
                background.material.depth_write = False
                scene.add(background)

            colors = np.asarray(colors, dtype=np.float32)
            if as_points:
                points = lines.get_data().astype(np.float32)
                scene.add(actor.point(points, colors=colors))
            elif no_tube:
                line_actor = actor.line(lines, colors=colors)
                # FURY lines with a color per point do not use their opacity
                line_actor.material.opacity = opacity
                scene.add(line_actor)
            else:
                thickness = linewidth
                if centroid_only or centroid_with_bundle:
                    thickness = 35 if centroid_only else 30
                scene.add(
                    actor.streamlines(
                        lines, colors=colors, thickness=thickness, opacity=opacity
                    )
                )

        for name in glass_brain or []:
            if name == "default_glass":
                template = read_mni_template(version="c", contrast="mask")
                data, affine = np.asarray(template.dataobj), template.affine
                scene.add(_contour(data, affine, (0, 0, 0), 0.04))
            elif name.endswith((".nii", ".nii.gz")):
                data, affine = load_nifti(name)
                scene.add(_contour(data, affine, (0, 0, 0), 0.04))
            elif name.endswith((".gii", ".gii.gz")):
                vertices, faces = load_gifti(name)
                scene.add(
                    actor.surface(
                        np.asarray(vertices, dtype=np.float32),
                        np.asarray(faces, dtype=np.int32),
                        colors=np.zeros((len(vertices), 3), dtype=np.float32),
                        opacity=0.05,
                    )
                )
            else:
                raise ValueError(f"Unknown glass brain {name}")

        mask_colors = list(mask_colors or [])
        for j, mask_file in enumerate(masks or []):
            color = (
                matplotlib.colors.to_rgb(mask_colors[j])
                if j < len(mask_colors)
                else (0, 0, 0)
            )
            data, affine = load_nifti(mask_file)
            scene.add(_contour(data, affine, color, mask_opacity))

        if show_bundle:
            show_m = window.ShowManager(scene=scene, size=(1200, 1200), title="SPECTRA")
            view_dir, up = cameras[cam_settings[0]]
            show_m.screens[0].camera.show_object(
                scene.main_scene, view_dir=view_dir, up=up
            )
            show_m.start()

        show_m = window.ShowManager(
            scene=scene, size=(1600, 1600), window_type="offscreen"
        )
        for k, name in enumerate(cam_settings):
            view_dir, up = cameras[name]
            show_m.screens[0].camera.show_object(
                scene.main_scene, view_dir=view_dir, up=up
            )
            show_m.render()
            show_m.window.draw()
            fname = Path(out_dir) / out_image
            if len(cam_settings) > 1:
                fname = fname.with_name(f"{fname.stem}_cam{k + 1:02d}{fname.suffix}")
            show_m.snapshot(fname=str(fname))
            if auto_crop:
                _autocrop(fname)
            logger.info(f"Snapshot saved as {fname}")
