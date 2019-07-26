import os

from dipy.io.image import save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import create_nifti_header
from dipy.tracking.streamline import Streamlines
from dipy.testing.decorators import xvfb_it, use_xvfb
from dipy.utils.optpkg import optional_package
from nibabel.tmpdirs import TemporaryDirectory
import numpy as np
import numpy.testing as npt

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from dipy.workflows.viz import HorizonFlow
    from dipy.viz.app import horizon


skip_it = use_xvfb == 'skip'


@npt.dec.skipif(skip_it or not has_fury)
@xvfb_it
def test_horizon_flow():

    s1 = 10 * np.array([[0, 0, 0],
                        [1, 0, 0],
                        [2, 0, 0],
                        [3, 0, 0],
                        [4, 0, 0]], dtype='f8')

    s2 = 10 * np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 2, 0],
                        [0, 3, 0],
                        [0, 4, 0]], dtype='f8')

    s3 = 10 * np.array([[0, 0, 0],
                        [1, 0.2, 0],
                        [2, 0.2, 0],
                        [3, 0.2, 0],
                        [4, 0.2, 0]], dtype='f8')

    print(s1.shape)
    print(s2.shape)
    print(s3.shape)

    streamlines = Streamlines()
    streamlines.append(s1)
    streamlines.append(s2)
    streamlines.append(s3)

    tractograms = [streamlines]
    images = None

    horizon(tractograms, images=images, cluster=True, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=False, interactive=False)
#
    affine = np.diag([2., 1, 1, 1]).astype('f8')
#
    data = 255 * np.random.rand(150, 150, 150)
#
    images = [(data, affine)]

    horizon(tractograms, images=images, cluster=True, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=True, interactive=False)

    with TemporaryDirectory() as out_dir:

        fimg = os.path.join(out_dir, 'test.nii.gz')
        ftrk = os.path.join(out_dir, 'test.trk')

        save_nifti(fimg, data, affine)
        dimensions = data.shape
        voxel_sizes = np.array([2, 1.0, 1.0])
        nii_header = create_nifti_header(affine, dimensions, voxel_sizes)
        sft = StatefulTractogram(streamlines, nii_header, space=Space.RASMM)
        save_tractogram(sft, ftrk, bbox_valid_check=False)

        input_files = [ftrk, fimg]

        npt.assert_equal(len(input_files), 2)

        hz_flow = HorizonFlow()

        hz_flow.run(input_files=input_files, stealth=True,
                    out_dir=out_dir, out_stealth_png='tmp_x.png')

        npt.assert_equal(os.path.exists(os.path.join(out_dir, 'tmp_x.png')),
                         True)


if __name__ == '__main__':

    test_horizon_flow()
