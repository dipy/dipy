import numpy as np
from dipy.data import get_data
from dipy.segment.mrf_map import (FASTImageSegmenter,
                                  initialize_constant_models_uniform,
                                  neg_log_likelihood_gaussian)


def test_segmentation():
    # Load a coronal slice from a T1-weighted MRI
    fname = get_data('t1_coronal_slice')
    single_slice = np.load(fname)
    # Stack a few copies to form a 3D volume
    nslices = 5
    image = np.zeros(shape=single_slice.shape + (nslices,))
    image[..., :nslices] = single_slice[..., None]
    # Execute the segmentation
    num_classes = 4
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


def test_initialization():

    fname = get_data('t1_coronal_slice')
    single_slice = np.load(fname)
    nslices = 5
    image = np.zeros(shape=single_slice.shape + (nslices,))
    image[..., :nslices] = single_slice[..., None]

    num_classes = 3

    mu, sigma = initialize_constant_models_uniform(image, num_classes)

    print(mu)
    print(sigma)


if __name__ == '__main__':

    test_initialization()
    #test_segmentation()