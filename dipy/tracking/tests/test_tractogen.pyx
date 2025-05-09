import nibabel as nib
import numpy as np
import numpy.testing as npt
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr

from dipy.core.sphere import HemiSphere
from dipy.data import get_fnames, get_sphere
from dipy.direction.peaks import peaks_from_positions
from dipy.direction.pmf import SimplePmfGen, SHCoeffPmfGen
from dipy.reconst.shm import sh_to_sf
from dipy.tracking.tractogen import generate_tractogram
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker_parameters import generate_tracking_parameters
from dipy.tracking.utils import (connectivity_matrix, random_seeds_from_mask,
                                 seeds_from_mask, seeds_directions_pairs)



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
    fnames = get_fnames(name="disco1", include_optional=True)

    # prepare ODFs
    if sphere is None:
        sphere = HemiSphere.from_sphere(get_sphere(name="repulsion724"))
    sh = nib.load(fnames[20]).get_fdata()
    fODFs = sh_to_sf(
        sh, sphere, sh_order_max=12, basis_type='tournier07', legacy=False
        )
    fODFs[fODFs<0] = 0
    pmf_gen = SimplePmfGen(np.asarray(fODFs, dtype=float), sphere)

    # seeds position and initial directions
    mask = nib.load(fnames[25]).get_fdata()
    affine = nib.load(fnames[25]).affine
    seed_mask = nib.load(fnames[34]).get_fdata()
    seed_mask = binary_erosion(seed_mask * mask, iterations=1)
    seed_positions = seeds_from_mask(seed_mask, affine, density=1)[:nbr_seeds]
    peaks = peaks_from_positions(seed_positions, fODFs, sphere, npeaks=1, affine=affine)
    seeds, directions = seeds_directions_pairs(seed_positions, peaks, max_cross=1)

    # stopping criterion
    sc = BinaryStoppingCriterion(mask)

    streamlines = generate_tractogram(seeds,
                                      directions,
                                      sc,
                                      params,
                                      pmf_gen,
                                      affine=affine,
                                      nbr_threads=nbr_threads)
    return streamlines


def get_disco_performances(streamlines):
    """
    Return the streamlines connectivity performance compared to the GT DiSCo connectom.
    """
    # fetch the disco data
    fnames = get_fnames(name="disco1", include_optional=True)

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


def test_return_all():
    """This tests that the number of streamlines equals the number of seeds
    when return_all=True.
    """


    fnames = get_fnames(name="disco1", include_optional=True)
    sphere = HemiSphere.from_sphere(get_sphere(name="repulsion724"))
    sh = nib.load(fnames[20]).get_fdata()
    fODFs = sh_to_sf(sh, sphere, sh_order_max=12, basis_type='tournier07', legacy=False)
    fODFs[fODFs<0] = 0
    pmf_gen = SimplePmfGen(np.asarray(fODFs, dtype=float), sphere)

    # seeds position and initial directions
    mask = nib.load(fnames[25]).get_fdata()
    sc = BinaryStoppingCriterion(mask)
    affine = nib.load(fnames[25]).affine
    seed_mask = np.ones(mask.shape)
    seeds = random_seeds_from_mask(seed_mask, affine, seeds_count=100,
                                   seed_count_per_voxel=False)
    directions = np.random.random(seeds.shape)
    directions = np.array([v/np.linalg.norm(v) for v in directions])

    # test return_all=True
    params = generate_tracking_parameters("det",
                                          max_len=500,
                                          min_len=0,
                                          step_size=0.5,
                                          voxel_size=np.ones(3),
                                          max_angle=20,
                                          random_seed=0,
                                          return_all=True)
    stream_gen = generate_tractogram(seeds,
                                     directions,
                                     sc,
                                     params,
                                     pmf_gen,
                                     affine=affine)
    streamlines = Streamlines(stream_gen)
    npt.assert_equal(len(streamlines), len(seeds))

    # test return_all=False
    params = generate_tracking_parameters("det",
                                          max_len=500,
                                          min_len=10,
                                          step_size=0.5,
                                          voxel_size=np.ones(3),
                                          max_angle=20,
                                          random_seed=0,
                                          return_all=False)
    stream_gen = generate_tractogram(seeds,
                                     directions,
                                     sc,
                                     params,
                                     pmf_gen,
                                     affine=affine)
    streamlines = Streamlines(stream_gen)
    npt.assert_array_less(len(streamlines), len(seeds))


def test_max_min_length():
    """This tests that the returned streamlines respect the length criterion.
    """
    fnames = get_fnames(name="disco1", include_optional=True)
    sphere = HemiSphere.from_sphere(get_sphere(name="repulsion724"))
    sh = nib.load(fnames[20]).get_fdata()
    fODFs = sh_to_sf(
        sh, sphere, sh_order_max=12, basis_type='tournier07', legacy=False
        )
    fODFs[fODFs<0] = 0
    pmf_gen = SimplePmfGen(np.asarray(fODFs, dtype=float), sphere)

    # seeds position and initial directions
    mask = nib.load(fnames[25]).get_fdata()
    sc = BinaryStoppingCriterion(mask)
    affine = nib.load(fnames[25]).affine
    seed_mask = np.ones(mask.shape)
    seeds = random_seeds_from_mask(seed_mask, affine, seeds_count=1000,
                                   seed_count_per_voxel=False)
    directions = np.random.random(seeds.shape)
    directions = np.array([v/np.linalg.norm(v) for v in directions])

    min_len=10
    max_len=100

    params = generate_tracking_parameters("det",
                                          max_len=max_len,
                                          min_len=min_len,
                                          step_size=0.5,
                                          voxel_size=np.ones(3),
                                          max_angle=20,
                                          random_seed=0,
                                          return_all=False)
    stream_gen = generate_tractogram(seeds,
                                     directions,
                                     sc,
                                     params,
                                     pmf_gen,
                                     affine=affine)
    streamlines = Streamlines(stream_gen)
    errors = np.array([len(s) < min_len and len(s) > max_len for s in streamlines])

    npt.assert_(np.sum(errors) == 0)


def test_buffer_frac():
    """This tests that the buffer fraction for generate tractogram plays well.
    """
    fnames = get_fnames(name="disco1", include_optional=True)
    sphere = HemiSphere.from_sphere(get_sphere(name="repulsion724"))
    sh = nib.load(fnames[20]).get_fdata()
    fODFs = sh_to_sf(
        sh, sphere, sh_order_max=12, basis_type='tournier07', legacy=False
        )
    fODFs[fODFs<0] = 0
    pmf_gen = SimplePmfGen(np.asarray(fODFs, dtype=float), sphere)

    # seeds position and initial directions
    mask = nib.load(fnames[25]).get_fdata()
    sc = BinaryStoppingCriterion(mask)
    affine = nib.load(fnames[25]).affine
    seed_mask = np.ones(mask.shape)
    seeds = random_seeds_from_mask(seed_mask, affine, seeds_count=500,
                                   seed_count_per_voxel=False)
    directions = np.random.random(seeds.shape)
    directions = np.array([v/np.linalg.norm(v) for v in directions])

    params = generate_tracking_parameters("prob",
                                          max_len=500,
                                          min_len=0,
                                          step_size=0.5,
                                          voxel_size=np.ones(3),
                                          max_angle=20,
                                          random_seed=0,
                                          return_all=True)

    streams = Streamlines(generate_tractogram(seeds,
                                              directions,
                                              sc,
                                              params,
                                              pmf_gen,
                                              affine=affine,
                                              buffer_frac=1.0))

    # test the results are identical with various buffer fractions
    for frac in [0.01,0.1,0.5]:
        frac_streams = Streamlines(generate_tractogram(seeds,
                                                       directions,
                                                       sc,
                                                       params,
                                                       pmf_gen,
                                                       affine=affine,
                                                       buffer_frac=frac))
        npt.assert_equal(len(frac_streams), len(streams))
