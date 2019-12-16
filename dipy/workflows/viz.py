import numpy as np
from os.path import join as pjoin
from dipy.workflows.workflow import Workflow
from dipy.io.image import load_nifti
from dipy.viz.app import horizon
from dipy.io.peaks import load_peaks
from dipy.io.streamline import load_tractogram
from dipy.io.utils import create_nifti_header


class HorizonFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'horizon'

    def run(self, input_files, cluster=False, cluster_thr=15.,
            random_colors=False, length_gt=0, length_lt=1000,
            clusters_gt=0, clusters_lt=10**8, native_coords=False,
            stealth=False, emergency_header='icbm_2009a', out_dir='',
            out_stealth_png='tmp.png'):
        """ Interactive medical visualization - Invert the Horizon!

        Interact with any number of .trk, .tck or .dpy tractograms and anatomy
        files .nii or .nii.gz. Cluster streamlines on loading.

        Parameters
        ----------
        input_files : variable string
        cluster : bool
            Enable QuickBundlesX clustering
        cluster_thr : float
            Distance threshold used for clustering. Default value 15.0 for
            small animal brains you may need to use something smaller such
            as 2.0. The distance is in mm. For this parameter to be active
            ``cluster`` should be enabled
        random_colors : bool
            Given multiple tractograms have been included then each tractogram
            will be shown with different color
        length_gt : float
            Clusters with average length greater than ``length_gt`` amount
            in mm will be shown
        length_lt : float
            Clusters with average length less than ``length_lt`` amount in
            mm will be shown
        clusters_gt : int
            Clusters with size greater than ``clusters_gt`` will be shown.
        clusters_lt : int
            Clusters with size less than ``clusters_gt`` will be shown.
        native_coords : bool
            Show results in native coordinates
        stealth : bool
            Do not use interactive mode just save figure.
        emergency_header : str
            If no anatomy reference is provided an emergency header is
            provided. Current options 'icbm_2009a' and 'icbm_2009c'.
        out_dir : string
            Output directory. Default current directory.
        out_stealth_png : string
            Filename of saved picture.

        References
        ----------
        .. [Horizon_ISMRM19] Garyfallidis E., M-A. Cote, B.Q. Chandio,
            S. Fadnavis, J. Guaje, R. Aggarwal, E. St-Onge, K.S. Juneja,
            S. Koudoro, D. Reagan, DIPY Horizon: fast, modular, unified and
            adaptive visualization, Proceedings of: International Society of
            Magnetic Resonance in Medicine (ISMRM), Montreal, Canada, 2019.
        """
        verbose = True
        tractograms = []
        images = []
        pams = []
        interactive = not stealth
        world_coords = not native_coords

        mni_2009a = {}
        mni_2009a['affine'] = np.array([[1., 0., 0., -98.],
                                        [0., 1., 0., -134.],
                                        [0., 0., 1., -72.],
                                        [0., 0., 0., 1.]])
        mni_2009a['dims'] = (197, 233, 189)
        mni_2009a['vox_size'] = (1., 1., 1.)
        mni_2009a['vox_space'] = 'RAS'

        mni_2009c = {}
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

            if ends('.trk'):

                sft = load_tractogram(fname, 'same',
                                      bbox_valid_check=False)
                tractograms.append(sft)

            if ends('.dpy') or ends('.tck'):
                sft = load_tractogram(fname, emergency_ref)
                tractograms.append(sft)

            if ends('.nii.gz') or ends('.nii'):

                data, affine = load_nifti(fname)
                images.append((data, affine))
                if verbose:
                    print('Affine to RAS')
                    np.set_printoptions(3, suppress=True)
                    print(affine)
                    np.set_printoptions()

            if ends(".pam5"):

                pam = load_peaks(fname)
                pams.append(pam)

                if verbose:
                    print('Peak_dirs shape')
                    print(pam.peak_dirs.shape)

        horizon(tractograms=tractograms, images=images, pams=pams,
                cluster=cluster, cluster_thr=cluster_thr,
                random_colors=random_colors,
                length_gt=length_gt, length_lt=length_lt,
                clusters_gt=clusters_gt, clusters_lt=clusters_lt,
                world_coords=world_coords,
                interactive=interactive,
                out_png=pjoin(out_dir, out_stealth_png))
