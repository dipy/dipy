import numpy as np
import numpy.testing as npt
from dipy.data import get_data
from dipy.segment.mrf_map import (FASTImageSegmenter,
                                  initialize_constant_models_uniform,
                                  neg_log_likelihood_gaussian)


# Load a coronal slice from a T1-weighted MRI
fname = get_data('t1_coronal_slice')
single_slice = np.load(fname)

# Stack a few copies to form a 3D volume
nslices = 5
image = np.zeros(shape=single_slice.shape + (nslices,))
image[..., :nslices] = single_slice[..., None]

# Execute the segmentation
num_classes = 4


def test_segmentation():

    beta = 0.01
    max_iter = 5
    segmenter = FASTImageSegmenter()
    segmented = segmenter.segment(image, num_classes, beta, max_iter)

    imshow(image[..., 0])
    figure()
    imshow(segmented[..., 0])

    # TO-DO: assert against a good segmentation
    # For now we can simply visualize the result and verify that
    # the segmentation gets smoother as we increase beta


def test_in_parts():

    mu, sigma = initialize_constant_models_uniform(image, num_classes)

    print(mu)
    print(sigma)

    npt.assert_almost_equal(mu, np.array([0., 0.25, 0.5, 0.75]))

    sigmasq = sigma ** 2

    neg_logl = neg_log_likelihood_gaussian(image, mu, sigmasq)

    print(neg_logl.shape)
    print(neg_logl.min())
    print(neg_logl.max())

    # add test here




if __name__ == '__main__':

    neg_logl = test_in_parts()
    #test_segmentation()