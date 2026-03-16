from dipy.testing.decorators import warning_for_keywords
from dipy.utils.deprecator import deprecate_with_version
from dipy.viz.skyline.app import skyline


@deprecate_with_version(
    (
        "Horizon is deprecated and will be removed in future releases. "
        "Please use Skyline instead."
    ),
    since="1.13.0",
    until="2.0.0",
)
class Horizon:
    @warning_for_keywords()
    def __init__(
        self,
        *,
        tractograms=None,
        images=None,
        pams=None,
        surfaces=None,
        cluster=False,
        rgb=False,
        cluster_thr=15.0,
        random_colors=None,
        length_gt=0,
        length_lt=1000,
        clusters_gt=0,
        clusters_lt=10000,
        world_coords=True,
        interactive=True,
        out_png="tmp.png",
        recorded_events=None,
        return_showm=False,
        bg_color=(0, 0, 0),
        order_transparent=True,
        buan=False,
        buan_colors=None,
        roi_images=False,
        roi_colors=(1, 0, 0),
        surface_colors=((1, 0, 0),),
    ):
        """Orchestrate interactive medical visualization for multimodal data.

        Invert the Horizon! The Horizon class acts as a central controller for
        visualizing medical imaging data, including tractograms, volumetric images,
        and surfaces. It manages the 3D scene, user interactions, and optional
        processing pipelines such as QuickBundlesX clustering
        :footcite:p:`Garyfallidis2019`.

        Parameters
        ----------
        tractograms : sequence of StatefulTractograms, optional
            StatefulTractograms are used for making sure that the coordinate
            systems are correct
        images : sequence of tuples, optional
            Each tuple contains data and affine
        pams : sequence of PeakAndMetrics, optional
            Contains peak directions and spherical harmonic coefficients
        surfaces : sequence of tuples, optional
            Each tuple contains vertices and faces
        cluster : bool, optional
            If True, applies QuickBundlesX clustering to the input tractograms
            for real-time simplification and grouping of streamlines.
        rgb : bool, optional
            Enable the color image (rgb only, alpha channel will be ignored).
        cluster_thr : float, optional
            Distance threshold used for clustering. Default value 15.0 for
            small animal data you may need to use something smaller such
            as 2.0. The threshold is in mm. For this parameter to be active
            ``cluster`` should be enabled.
        random_colors : string, optional
            Given multiple tractograms and/or ROIs then each tractogram and/or
            ROI will be shown with a different color. If no value is provided,
            both the tractograms and the ROIs will have a different random
            color generated from a distinguishable colormap. If the effect
            should only be applied to one of the 2 types, then use the
            options 'tracts' and 'rois' for the tractograms and the ROIs
            respectively.
        length_gt : float, optional
            Clusters with average length greater than ``length_gt`` amount
            in mm will be shown.
        length_lt : float, optional
            Clusters with average length less than ``length_lt`` amount in mm
            will be shown.
        clusters_gt : int, optional
            Clusters with size greater than ``clusters_gt`` will be shown.
        clusters_lt : int, optional
            Clusters with size less than ``clusters_lt`` will be shown.
        world_coords : bool, optional
            Show data in their world coordinates (not native voxel coordinates)
            Default True.
        interactive : bool, optional
            Allow user interaction. If False then Horizon goes on stealth mode
            and just saves pictures.
        out_png : string, optional
            Filename of saved picture.
        recorded_events : string, optional
            File path to replay recorded events
        return_showm : bool, optional
            Return ShowManager object. Used only at Python level. Can be used
            for extending Horizon's capabilities externally and for testing
            purposes.
        bg_color : ndarray or list or tuple, optional
            Define the background color of the scene.
            Default is black (0, 0, 0)
        order_transparent : bool, optional
            Default True. Use depth peeling to sort transparent objects.
            If True also enables anti-aliasing.
        buan : bool, optional
            Enables BUAN framework visualization. Default is False.
        buan_colors : list, optional
            List of colors for bundles.
        roi_images : bool, optional
            Displays binary images as contours. Default is False.
        roi_colors : ndarray or list or tuple, optional
            Define the colors of the roi images. Default is red (1, 0, 0)


        References
        ----------
        .. footbibliography::
        """
        self.show_m = None
        self.visualizations = []
        self._tractogram_help = False

        self._horizon_data = {
            "visualizer_type": "stealth" if not interactive else "standalone",
            "images": images,
            "tractograms": tractograms,
            "pams": pams,
            "surfaces": surfaces,
            "cluster": cluster,
            "rgb": rgb,
            "bg_color": bg_color,
            "cluster_thr": cluster_thr,
            "cluster_length_thr": length_gt,
            "cluster_size_thr": clusters_gt,
            "out_stealth_png": out_png,
            "interactive": interactive,
        }

    def build_show(self):
        skyline_window = skyline(
            visualizer_type="stealth"
            if not self._horizon_data["interactive"]
            else "standalone",
            images=self._horizon_data["images"],
            tractograms=self._horizon_data["tractograms"],
            peaks=self._horizon_data["pams"],
            surfaces=self._horizon_data["surfaces"],
            is_cluster=self._horizon_data["cluster"],
            rgb=self._horizon_data["rgb"],
            bg_color=self._horizon_data["bg_color"],
            cluster_thr=self._horizon_data["cluster_thr"],
            cluster_length_thr=self._horizon_data["cluster_length_thr"],
            cluster_size_thr=self._horizon_data["cluster_size_thr"],
            out_stealth_png=self._horizon_data["out_stealth_png"],
        )
        self.show_m = skyline_window.window
        return self.show_m


@deprecate_with_version(
    (
        "Horizon is deprecated and will be removed in future releases. "
        "Please use Skyline instead."
    ),
    since="1.13.0",
    until="2.0.0",
)
@warning_for_keywords()
def horizon(
    *,
    tractograms=None,
    images=None,
    pams=None,
    surfaces=None,
    cluster=False,
    rgb=False,
    cluster_thr=15.0,
    random_colors=None,
    bg_color=(0, 0, 0),
    order_transparent=True,
    length_gt=0,
    length_lt=1000,
    clusters_gt=0,
    clusters_lt=10000,
    world_coords=True,
    interactive=True,
    buan=False,
    buan_colors=None,
    roi_images=False,
    roi_colors=(1, 0, 0),
    out_png="tmp.png",
    recorded_events=None,
    return_showm=False,
):
    """Horizon is deprecated and will be removed with future releases.
    Please use Skyline.

    See :footcite:p:`Garyfallidis2019` for further details about Horizon.

    Parameters
    ----------
    tractograms : sequence of StatefulTractograms
        StatefulTractograms are used for making sure that the coordinate
        systems are correct
    images : sequence of tuples
        Each tuple contains data and affine
    pams : sequence of PeakAndMetrics
        Contains peak directions and spherical harmonic coefficients
    surfaces : sequence of tuples
        Each tuple contains vertices and faces
    cluster : bool
        Enable QuickBundlesX clustering
    rgb: bool, optional
        Enable the color image.
    cluster_thr : float
        Distance threshold used for clustering. Default value 15.0 for
        small animal data you may need to use something smaller such
        as 2.0. The threshold is in mm. For this parameter to be active
        ``cluster`` should be enabled.
    random_colors : string
        Given multiple tractograms and/or ROIs then each tractogram and/or
        ROI will be shown with different color. If no value is provided both
        the tractograms and the ROIs will have a different random color
        generated from a distinguishable colormap. If the effect should only be
        applied to one of the 2 objects, then use the options 'tracts' and
        'rois' for the tractograms and the ROIs respectively.
    bg_color : ndarray or list or tuple
        Define the background color of the scene. Default is black (0, 0, 0)
    order_transparent : bool
        Default True. Use depth peeling to sort transparent objects.
        If True also enables anti-aliasing.
    length_gt : float
        Clusters with average length greater than ``length_gt`` amount
        in mm will be shown.
    length_lt : float
        Clusters with average length less than ``length_lt`` amount in mm
        will be shown.
    clusters_gt : int
        Clusters with size greater than ``clusters_gt`` will be shown.
    clusters_lt : int
        Clusters with size less than ``clusters_lt`` will be shown.
    world_coords : bool
        Show data in their world coordinates (not native voxel coordinates)
        Default True.
    interactive : bool
        Allow user interaction. If False then Horizon goes on stealth mode
        and just saves pictures.
    buan : bool, optional
        Enables BUAN framework visualization. Default is False.
    buan_colors : list, optional
        List of colors for bundles.
    roi_images : bool, optional
        Displays binary images as contours. Default is False.
    roi_colors : ndarray or list or tuple, optional
        Define the color of the roi images. Default is red (1, 0, 0)
    out_png : string
        Filename of saved picture.
    recorded_events : string
        File path to replay recorded events
    return_showm : bool
        Return ShowManager object. Used only at Python level. Can be used
        for extending Horizon's capabilities externally and for testing
        purposes.
        This is ignored in Skyline.

    References
    ----------
    .. footbibliography::
    """

    hz = Horizon(
        tractograms=tractograms,
        images=images,
        pams=pams,
        surfaces=surfaces,
        cluster=cluster,
        rgb=rgb,
        cluster_thr=cluster_thr,
        random_colors=random_colors,
        length_gt=length_gt,
        length_lt=length_lt,
        clusters_gt=clusters_gt,
        clusters_lt=clusters_lt,
        world_coords=world_coords,
        interactive=interactive,
        out_png=out_png,
        recorded_events=recorded_events,
        return_showm=return_showm,
        bg_color=bg_color,
        order_transparent=order_transparent,
        buan=buan,
        buan_colors=buan_colors,
        roi_images=roi_images,
        roi_colors=roi_colors,
    )

    return hz.build_show()
