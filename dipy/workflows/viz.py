from os.path import join as pjoin
from dipy.workflows.workflow import Workflow
from dipy.io.streamline import load_tractogram
from dipy.io.image import load_nifti
from dipy.viz.app import horizon


class HorizonFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'horizon'

    def run(self, input_files, cluster=False, cluster_thr=15.,
            random_colors=False, length_lt=1000, length_gt=0,
            clusters_lt=10**8, clusters_gt=0, native_coords=False,
            stealth=False, out_dir='', out_stealth_png='tmp.png'):

        """ Highly interactive visualization - invert the Horizon!

        Interact with any number of .trk, .tck or .dpy tractograms and anatomy
        files .nii or .nii.gz. Cluster streamlines on loading.

        Parameters
        ----------
        input_files : variable string
        cluster : bool
        cluster_thr : float
        random_colors : bool
        length_lt : float
        length_gt : float
        clusters_lt : int
        clusters_gt : int
        native_coords : bool
        stealth : bool
        out_dir : string
        out_stealth_png : string


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
        interactive = not stealth
        world_coords = not native_coords

        io_it = self.get_io_iterator()

        for input_output in io_it:

            fname = input_output[0]

            if verbose:
                print('Loading file ...')
                print(fname)
                print('\n')

            fl = fname.lower()
            ends = fl.endswith

            if ends('.trk') or ends('.tck') or ends('.dpy'):

                streamlines, hdr = load_tractogram(fname, lazy_load=False)
                tractograms.append(streamlines)

            if ends('.nii.gz') or ends('.nii'):

                data, affine = load_nifti(fname)
                images.append((data, affine))
                if verbose:
                    print(affine)

        horizon(tractograms, images, cluster, cluster_thr,
                random_colors, length_lt, length_gt, clusters_lt,
                clusters_gt,
                world_coords=world_coords,
                interactive=interactive,
                out_png=pjoin(out_dir, out_stealth_png))
