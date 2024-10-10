import nibabel as nib
import numpy as np
import numpy.testing as npt
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr

from dipy.core.sphere import HemiSphere
from dipy.data import get_fnames, get_sphere
from dipy.direction.peaks import peaks_from_positions
from dipy.direction.pmf import SimplePmfGen
from dipy.reconst.shm import sh_to_sf
from dipy.tracking.fast_tracking import generate_tractogram
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.tracker_parameters import generate_tracking_parameters
from dipy.tracking.utils import (connectivity_matrix, seeds_from_mask,
                                 seeds_directions_pairs)


def get_fast_tracking_performances(params, *, nbr_seeds=1000, nbr_threads=0):
    """
    Generate streamlines and return performances on the DiSCo dataset.
    """
    s = generate_disco_streamlines(params, nbr_seeds=nbr_seeds, nbr_threads=nbr_threads)
    return get_disco_performances(s)


def generate_disco_streamlines(params, *, nbr_seeds=1000, nbr_threads=0, sphere=None):
    """
    Return streamlines generated on the DiSCo dataset 
    using the fast tracking module with the input tracking params 
    """
    # fetch the disco data
    fnames = get_fnames(name="disco1")
    
    # prepare ODFs
    if sphere is None:
        sphere = HemiSphere.from_sphere(get_sphere(name="repulsion724"))
    GT_SH = nib.load(fnames[20]).get_fdata()
    GT_ODF = sh_to_sf(
        GT_SH, sphere, sh_order_max=12, basis_type='tournier07', legacy=False
        )
    GT_ODF[GT_ODF<0] = 0

    # seeds position and initial directions
    mask = nib.load(fnames[25]).get_fdata()
    affine = nib.load(fnames[25]).affine
    seed_mask = nib.load(fnames[34]).get_fdata()
    seed_mask = binary_erosion(seed_mask * mask, iterations=1)
    seeds_positions = seeds_from_mask(seed_mask, affine, density=1)[:nbr_seeds]

    pmf_gen = SimplePmfGen(np.asarray(GT_ODF, dtype=float), sphere)
    peaks = peaks_from_positions(
        seeds_positions, GT_ODF, sphere, npeaks=1, affine=affine
        )
    seeds, initial_directions = seeds_directions_pairs(
        seeds_positions,
        peaks,
        max_cross=1
    )

    # stopping criterion
    sc = BinaryStoppingCriterion(mask)

    streamlines = generate_tractogram(seeds,
                                      initial_directions,
                                      sc,
                                      params,
                                      pmf_gen,
                                      nbr_threads=nbr_threads)
    return streamlines


def get_disco_performances(streamlines):
    """
    Return the streamlines connectivity performance compared to the GT DiSCo connectom.
    """
    # fetch the disco data
    fnames = get_fnames(name="disco1")    

    # prepare the GT connectome data
    GT_connectome = np.loadtxt(fnames[35])
    connectome_mask = np.tril(np.ones(GT_connectome.shape), -1) > 0
    labels_img = nib.load(fnames[23])
    affine = labels_img.affine
    labels = np.round(labels_img.get_fdata()).astype(int)
    connectome = connectivity_matrix(streamlines, affine, labels)[1:,1:]

    r, _ = pearsonr(GT_connectome[connectome_mask].flatten(),
                    connectome[connectome_mask].flatten())
    return r


def test_tractogram_reproducibility():
    # Test tractogram reproducibility
    params0 = generate_tracking_parameters("prob",
                                           max_len=500,
                                           step_size=0.2,
                                           voxel_size=np.ones(3),
                                           max_angle=20,
                                           random_seed=0)  

    params1 = generate_tracking_parameters("prob",
                                           max_len=500,
                                           step_size=0.2,
                                           voxel_size=np.ones(3),
                                           max_angle=20,
                                           random_seed=1)

    params2 = generate_tracking_parameters("prob",
                                           max_len=500,
                                           step_size=0.2,
                                           voxel_size=np.ones(3),
                                           max_angle=20,
                                           random_seed=2)
    
    # Same random generator seed
    r1 = get_fast_tracking_performances(params1, nbr_seeds=100, nbr_threads=1)
    r2 = get_fast_tracking_performances(params1, nbr_seeds=100, nbr_threads=1)                                
    npt.assert_equal(r1, r2)

    # Random random generator seed
    r1 = get_fast_tracking_performances(params0, nbr_seeds=100, nbr_threads=1)
    r2 = get_fast_tracking_performances(params0, nbr_seeds=100, nbr_threads=1)
    npt.assert_(not r1 == r2, msg="Tractograms are identical (random seed).")

    # Different random generator seeds
    r1 = get_fast_tracking_performances(params1, nbr_seeds=100, nbr_threads=1)
    r2 = get_fast_tracking_performances(params2, nbr_seeds=100, nbr_threads=1)
    npt.assert_(not r1 == r2, msg="Tractograms are identical (different seeds).")

    # ValeError raise when using more then 1 thread
    npt.assert_raises(
        ValueError,
        get_fast_tracking_performances,
        params1,
        nbr_seeds=100,
        nbr_threads=2,
        )
    npt.assert_raises(
        ValueError,
        get_fast_tracking_performances,
        params1,
        nbr_seeds=100,
        nbr_threads=0,
        )


def test_tracking_max_angle():
    """This tests that the angle between streamline points is always smaller
    then the input `max_angle` parameter.
    """

    def get_min_cos_similarity(streamlines):
        min_cos_sim = 1
        for sl in streamlines:
            if len(sl) > 1:
                v = sl[:-1] - sl[1:]  # vectors have norm of 1
                for i in range(len(v) - 1):
                    cos_sim = np.dot(v[i], v[i + 1])
                    if cos_sim < min_cos_sim:
                        min_cos_sim = cos_sim
        return min_cos_sim
    
    for sph in [
        get_sphere(name="repulsion100"),
        HemiSphere.from_sphere(get_sphere(name="repulsion100")),
    ]:
        for max_angle in [20,45]:

            params = generate_tracking_parameters("det",
                                                max_len=500,
                                                step_size=1,
                                                voxel_size=np.ones(3),
                                                max_angle=max_angle,
                                                random_seed=0)

            streamlines = generate_disco_streamlines(params, nbr_seeds=100, sphere=sph)
            min_cos_sim = get_min_cos_similarity(streamlines)
            npt.assert_(np.arccos(min_cos_sim) <= np.deg2rad(max_angle))


def test_tracking_step_size():
    """This tests that the distance between streamline
    points is equal to the input `step_size` parameter.
    """
    
    def get_points_distance(streamlines):
        dists = []
        for sl in streamlines:
            dists.extend(np.linalg.norm(sl[0:-1] - sl[1:], axis=1))            
        return dists


    for step_size in [0.02, 0.5, 1]:
        params = generate_tracking_parameters("det",
                                            max_len=500,
                                            step_size=step_size,
                                            voxel_size=np.ones(3),
                                            max_angle=20,
                                            random_seed=0)

        streamlines = generate_disco_streamlines(params, nbr_seeds=100)
        dists = get_points_distance(streamlines)
        npt.assert_almost_equal(np.min(dists), step_size)
        npt.assert_almost_equal(np.max(dists), step_size)
