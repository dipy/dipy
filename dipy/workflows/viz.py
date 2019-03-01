from dipy.workflows.workflow import Workflow
from dipy.io.streamline import load_tractogram
from dipy.io.image import load_nifti
# from dipy.io.peaks import load_peaks
from dipy.viz.app import horizon


class HorizonFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'horizon'

    def run(self, input_files, cluster=False, cluster_thr=15.,
            random_colors=False, length_lt=1000, length_gt=0,
            clusters_lt=10**8, clusters_gt=0, native_coords=False,
            stealth=False, out_dir='', out_stealth_png='tmp.png'):

        """ Advanced visualization application

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
        """
        verbose = True
        tractograms = []
        images = []
        interactive = not stealth
        world_coords = not native_coords

        io_it = self.get_io_iterator()

        for input_output in io_it:

            f = input_output[0]

            if verbose:
                print('Loading file ...')
                print(f)
                print('\n')

            fl = f.lower()
            ends = fl.endswith

            if ends('.trk') or ends('.tck') or ends('.dpy'):

                streamlines, hdr = load_tractogram(f, lazy_load=False)
                tractograms.append(streamlines)

            if ends('.nii.gz') or ends('.nii'):

                data, affine = load_nifti(f)
                images.append((data, affine))
                if verbose:
                    print(affine)

            """ TODO: add support for peaks
            if ends('.pam5'):

                peaks = load_peaks(f)
                if verbose:
                    print(peaks.peak_dirs.shape)
            """
        horizon(tractograms, images, cluster, cluster_thr,
                random_colors, length_lt, length_gt, clusters_lt,
                clusters_gt,
                world_coords=world_coords,
                interactive=interactive)
