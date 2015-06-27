from dipy.data import get_data
from dipy.segment.mrf_map import FASTImageSegmenter

def test_segmentation():
    # Load a coronal slice from a T1-weighted MRI
    fname = get_data('t1_coronal_slice')
    slice = np.load(fname)
    # Stack a few copies to form a 3D volume
    nslices = 5
    image = np.zeros(shape=slice.shape + (nslices,))
    image[..., :nslices] = slice[..., None]
    # Execute the segmentation
    num_classes = 4
    beta = 0.01
    max_iter = 1
    segmenter = FASTImageSegmenter()
    segmented = segmenter.segment(image, num_classes, beta, max_iter)
    # TO-DO: assert against a good segmentation
    # For now we can simply visualize the result and verify that
    # the segmentation gets smoother as we increase beta
