from dipy.workflows.workflow import Workflow
from dipy.io.streamline import load_trk
from dipy.io.image import load_nifti
from dipy.viz.app import horizon


class HorizonFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'horizon'

    def run(self, input_files, cluster=False, cluster_thr=15.,
            random_colors=False,
            length_lt=1000, length_gt=0,
            clusters_lt=10**8, clusters_gt=0):
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
        """
        verbose = True
        tractograms = []
        images = []

        for f in input_files:

            if verbose:
                print('Loading file ...')
                print(f)
                print('\n')

            if f.endswith('.trk'):

                streamlines, hdr = load_trk(f)
                tractograms.append(streamlines)

            if f.endswith('.nii.gz') or f.endswith('.nii'):

                data, affine = load_nifti(f)
                images.append((data, affine))
                if verbose:
                    print(affine)

        horizon(tractograms, images, cluster, cluster_thr,
                random_colors, length_lt, length_gt, clusters_lt,
                clusters_gt)
