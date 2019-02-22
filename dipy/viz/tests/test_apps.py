import numpy as np
from dipy.viz.app import horizon
from dipy.tracking.streamline import Streamlines


def test_horizon():

    s1 = 100 * np.array([[0, 0, 0],
                         [1, 0, 0],
                         [2, 0, 0],
                         [3, 0, 0],
                         [4, 0, 0]], dtype='f8')

    s2 = 100 * np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 2, 0],
                         [0, 3, 0],
                         [0, 4, 0]], dtype='f8')

    print(s1.shape)
    print(s2.shape)

    streamlines = Streamlines()
    streamlines.append(s1)
    streamlines.append(s2)

    tractograms = [streamlines]
    images = None

    horizon(tractograms, images=images, cluster=True, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=False, interactive=True)


if __name__ == '__main__':

    test_horizon()
