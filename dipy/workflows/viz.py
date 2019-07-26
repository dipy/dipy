import numpy as np
from os.path import join as pjoin
import nibabel as nib
from dipy.workflows.workflow import Workflow
from dipy.io.streamline import Dpy
from dipy.io.image import load_nifti
from dipy.viz.app import horizon
from dipy.io.peaks import load_peaks


class HorizonFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'horizon'

    def run(self, input_files, cluster=False, cluster_thr=15.,
            random_colors=False, length_gt=0, length_lt=1000,
            clusters_gt=0, clusters_lt=10**8, native_coords=False,
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
        length_gt : float
        length_lt : float
        clusters_gt : int
        clusters_lt : int
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
        pams = []
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

            if ends('.trk') or ends('.tck'):

                streamlines = nib.streamlines.load(fname).streamlines
                tractograms.append(streamlines)
            elif ends('dpy'):

                dpy_obj = Dpy(fname, mode='r')
                streamlines = list(dpy_obj.read_tracks())
                dpy_obj.close()

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
