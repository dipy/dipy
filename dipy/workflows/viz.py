from warnings import warn
import numpy as np
from os.path import join as pjoin
from dipy.io.surface import load_gifti, load_pial
from dipy.workflows.workflow import Workflow
from dipy.io.image import load_nifti
from dipy.viz import horizon
from dipy.io.peaks import load_peaks
from dipy.io.streamline import load_tractogram
from dipy.io.utils import create_nifti_header
from dipy.stats.analysis import assignment_map
from dipy.utils.optpkg import optional_package


fury, has_fury, setup_module = optional_package('fury', min_version="0.10.0")


if has_fury:
    from fury.colormap import line_colors
    from fury.utils import numpy_to_vtk_colors
    from fury.lib import numpy_support


class HorizonFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'horizon'

    def run(self, input_files, cluster=False, rgb=False, cluster_thr=15.,
            random_colors=None, length_gt=0, length_lt=1000,
            clusters_gt=0, clusters_lt=10**8, native_coords=False,
            stealth=False, emergency_header='icbm_2009a', bg_color=(0, 0, 0),
            disable_order_transparency=False, buan=False, buan_thr=0.5,
            buan_highlight=(1, 0, 0), roi_images=False, roi_colors=(1, 0, 0),
            out_dir='', out_stealth_png='tmp.png'):
        """ Interactive medical visualization - Invert the Horizon!

        Interact with any number of .trk, .tck or .dpy tractograms and anatomy
        files .nii or .nii.gz. Cluster streamlines on loading.

        Parameters
        ----------
        input_files : variable string
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
        length_gt : float, optional
            Clusters with average length greater than ``length_gt`` amount
            in mm will be shown.
        length_lt : float, optional
            Clusters with average length less than ``length_lt`` amount in
            mm will be shown.
        clusters_gt : int, optional
            Clusters with size greater than ``clusters_gt`` will be shown.
        clusters_lt : int, optional
            Clusters with size less than ``clusters_gt`` will be shown.
        native_coords : bool, optional
            Show results in native coordinates.
        stealth : bool, optional
            Do not use interactive mode just save figure.
        emergency_header : str, optional
            If no anatomy reference is provided an emergency header is
            provided. Current options 'icbm_2009a' and 'icbm_2009c'.
        bg_color : variable float, optional
            Define the background color of the scene. Colors can be defined
            with 1 or 3 values and should be between [0-1].
        disable_order_transparency : bool, optional
            Use depth peeling to sort transparent objects.
            If True also enables anti-aliasing.
        buan : bool, optional
            Enables BUAN framework visualization.
        buan_thr : float, optional
            Uses the threshold value to highlight segments on the
            bundle which have pvalues less than this threshold.
        buan_highlight : variable float, optional
            Define the bundle highlight area color. Colors can be defined
            with 1 or 3 values and should be between [0-1].
            For example, a value of (1, 0, 0) would mean the red color.
        roi_images : bool, optional
            Displays binary images as contours.
        roi_colors : variable float, optional
            Define the color for the roi images. Colors can be defined
            with 1 or 3 values and should be between [0-1]. For example, a
            value of (1, 0, 0) would mean the red color.
        out_dir : str, optional
            Output directory. (default current directory)
        out_stealth_png : str, optional
            Filename of saved picture.

        References
        ----------
        .. [Horizon_ISMRM19] Garyfallidis E., M-A. Cote, B.Q. Chandio,
            S. Fadnavis, J. Guaje, R. Aggarwal, E. St-Onge, K.S. Juneja,
            S. Koudoro, D. Reagan, DIPY Horizon: fast, modular, unified and
            adaptive visualization, Proceedings of: International Society of
            Magnetic Resonance in Medicine (ISMRM), Montreal, Canada, 2019.
        """
        super(HorizonFlow, self).__init__(force=True)
        verbose = True
        tractograms = []
        images = []
        pams = []
        surfaces = []
        numpy_files = []
        interactive = not stealth
        world_coords = not native_coords
        bundle_colors = None

        mni_2009a = dict()
        mni_2009a['affine'] = np.array([[1., 0., 0., -98.],
                                        [0., 1., 0., -134.],
                                        [0., 0., 1., -72.],
                                        [0., 0., 0., 1.]])
        mni_2009a['dims'] = (197, 233, 189)
        mni_2009a['vox_size'] = (1., 1., 1.)
        mni_2009a['vox_space'] = 'RAS'

        mni_2009c = dict()
        mni_2009c['affine'] = np.array([[1., 0., 0., -96.],
                                        [0., 1., 0., -132.],
                                        [0., 0., 1., -78.],
                                        [0., 0., 0., 1.]])
        mni_2009c['dims'] = (193, 229, 193)
        mni_2009c['vox_size'] = (1., 1., 1.)
        mni_2009c['vox_space'] = 'RAS'

        if emergency_header == 'icbm_2009a':
            hdr = mni_2009c
        else:
            hdr = mni_2009c
        emergency_ref = create_nifti_header(hdr['affine'], hdr['dims'],
                                            hdr['vox_size'])

        io_it = self.get_io_iterator()

        for input_output in io_it:

            fname = input_output[0]

            if verbose:
                print('Loading file ...')
                print(fname)
                print('\n')

            fl = fname.lower()
            ends = fl.endswith

            if ends('.trk') or ends('.trx'):

                sft = load_tractogram(fname, 'same',
                                      bbox_valid_check=False)
                tractograms.append(sft)

            if ends('.dpy') or ends('.tck'):
                sft = load_tractogram(fname, emergency_ref)
                tractograms.append(sft)

            if ends('.nii.gz') or ends('.nii'):

                data, affine = load_nifti(fname)
                images.append((data, affine, fname))
                if verbose:
                    print('Affine to RAS')
                    np.set_printoptions(3, suppress=True)
                    print(affine)
                    np.set_printoptions()

            if ends('.pial'):
                surface = load_pial(fname)
                if surface:
                    vertices, faces = surface
                    surfaces.append((vertices, faces, fname))

            if ends('.gii.gz') or ends('.gii'):
                surface = load_gifti(fname)
                vertices, faces = surface
                if len(vertices) and len(faces):
                    vertices, faces = surface
                    surfaces.append((vertices, faces, fname))
                else:
                    warn(f'{fname} does not have any surface geometry.')

            if ends(".pam5"):

                pam = load_peaks(fname)
                pams.append(pam)

                if verbose:
                    print('Peak_dirs shape')
                    print(pam.peak_dirs.shape)

            if ends(".npy"):

                data = np.load(fname)
                numpy_files.append(data)

                if verbose:
                    print('numpy array length')
                    print(len(data))

        if buan:
            bundle_colors = []

            for i in range(len(numpy_files)):

                n = len(numpy_files[i])
                pvalues = numpy_files[i]
                bundle = tractograms[i].streamlines

                indx = assignment_map(bundle, bundle, n)
                ind = np.array(indx)

                nb_lines = len(bundle)
                lines_range = range(nb_lines)
                points_per_line = [len(bundle[i]) for i in lines_range]
                points_per_line = np.array(points_per_line, np.intp)

                cols_arr = line_colors(bundle)
                colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
                colors = numpy_support.vtk_to_numpy(vtk_colors)
                colors = (colors - np.min(colors))/np.ptp(colors)

                for j in range(n):

                    if pvalues[j] < buan_thr:
                        colors[ind == j] = buan_highlight

                bundle_colors.append(colors)

        if len(bg_color) == 1:
            bg_color *= 3
        elif len(bg_color) != 3:
            raise ValueError('You need 3 values to set up background color. '
                             'e.g --bg_color 0.5 0.5 0.5')

        if len(roi_colors) == 1:
            roi_colors *= 3
        elif len(roi_colors) != 3:
            raise ValueError('You need 3 values to set up ROI color. '
                             'e.g. --roi_colors 0.5 0.5 0.5')

        order_transparent = not disable_order_transparency
        horizon(tractograms=tractograms, images=images, pams=pams,
                surfaces=surfaces, cluster=cluster, rgb=rgb,
                cluster_thr=cluster_thr, random_colors=random_colors,
                bg_color=bg_color, order_transparent=order_transparent,
                length_gt=length_gt, length_lt=length_lt,
                clusters_gt=clusters_gt, clusters_lt=clusters_lt,
                world_coords=world_coords,
                interactive=interactive, buan=buan, buan_colors=bundle_colors,
                roi_images=roi_images, roi_colors=roi_colors,
                out_png=pjoin(out_dir, out_stealth_png))
