from dipy.io.utils import split_filename_extension
from dipy.utils.logging import logger
from dipy.viz import skyline_from_files
from dipy.workflows.workflow import Workflow


class SkylineFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "skyline"

    def run(
        self,
        input_files,
        *,
        rois=None,
        odfs=None,
        cluster=False,
        performance_version=False,
        glass_brain=False,
        bg_color=None,
        tract_colors=None,
        cluster_thr=15.0,
        cluster_size_thr=-1,
        cluster_length_thr=-1,
        buan_pvals=None,
        rgb=False,
        stealth=False,
        out_dir="",
        out_stealth_png="out_skyline.png",
    ):
        """Launch Skyline GUI.

        If you want to load only odfs or rois.
        use dipy_skyline run --odfs <Your ODF files> or
        dipy_skyline run --rois <Your ROI files> respectively.

        These options should be used in stealth mode. For GUI mode, the files can be
        loaded through the file dialog.


        Parameters
        ----------
        input_files : variable string or Path
            Tuple of path for each image, peak, surface or tractogram to be added to
            the Skyline viewer.
        rois : variable str, optional
            Tuple of path for each ROI to be added to the Skyline viewer.
        odfs : variable str, optional
            Tuple of path for each ODF to be added to the Skyline viewer.
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
            Do not use interactive mode just save figure.
        rgb : bool, optional
            Enable the colors in the image if 4D data with RGB/RGBA channels.
        out_dir : str or Path, optional
            Output directory to save the figure if stealth mode is enabled.
        out_stealth_png : str, optional
            Filename of saved picture if stealth mode is enabled.
        """
        super(SkylineFlow, self).__init__(force=True)
        skyline_input_files = []
        start_gui = input_files is not None and input_files[0] in (
            "run",
            "start",
            "launch",
            "initialize",
        )
        if not start_gui:
            io_it = self.get_io_iterator()
            for input_output in io_it:
                skyline_input_files.append(input_output[0])

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
            shm_coeffs=odfs,
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
        rgb=False,
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
        rgb : bool, optional
            Enable the color image (rgb only, alpha channel will be ignored).
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
        super(HorizonFlow, self).__init__(force=True)

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

        super(HorizonFlow, self).run(
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
