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
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker_parameters import generate_tracking_parameters
from dipy.tracking.utils import (connectivity_matrix, random_seeds_from_mask,
                                 seeds_directions_pairs)


def test_fast_tracking_performances():
    """
    This test the performance of the fast tracking module
    on the DiSCo dataset
    """
    # fetch the disco data
    fnames = get_fnames("disco1")

    # prepare the GT connectome data
    GT_connectome = np.loadtxt(fnames[35])
    connectome_mask = np.tril(np.ones(GT_connectome.shape), -1) > 0
    labels = np.round(nib.load(fnames[23]).get_fdata()).astype(int)  # 6 low res

    # prepare ODFs
    sphere = HemiSphere.from_sphere(get_sphere("repulsion724"))
    GT_SH = nib.load(fnames[20]).get_fdata()  # 17 low res
    GT_ODF = sh_to_sf(GT_SH, sphere, sh_order_max=12, basis_type='tournier07', legacy=False)
    GT_ODF[GT_ODF<0] = 0

    # seeds position and initial directions
    mask = nib.load(fnames[25]).get_fdata()  # 7 low res
    affine = nib.load(fnames[25]).affine  # 7 low res
    seed_mask = nib.load(fnames[34]).get_fdata()  # 9 low res
    seed_mask = binary_erosion(seed_mask * mask, iterations=1)
    seeds_positions = random_seeds_from_mask(seed_mask,
                                             affine,
                                             seeds_count=5000,
                                             seed_count_per_voxel=False)

    pmf_gen = SimplePmfGen(np.asarray(GT_ODF, dtype=float), sphere)
    peaks = peaks_from_positions(seeds_positions, GT_ODF, sphere, npeaks=1, affine=affine)
    seeds, initial_directions = seeds_directions_pairs(seeds_positions,
                                                       peaks,
                                                       max_cross=1)

    # stopping criterion
    sc = BinaryStoppingCriterion(mask)

    # TEST fast probabilistic tracking

    params_prob = generate_tracking_parameters("prob",
                                               max_len=500,
                                               step_size=0.2,
                                               voxel_size=np.ones(3),
                                               max_angle=20)
    params_det = generate_tracking_parameters("det",
                                              max_len=500,
                                              step_size=0.2,
                                              voxel_size=np.ones(3),
                                              max_angle=20)
    params_ptt = generate_tracking_parameters("ptt",
                                               max_len=500,
                                               step_size=0.2,
                                               voxel_size=np.ones(3),
                                               max_angle=15,
                                               probe_quality=4)
    for algo, params in [("prob", params_prob), ("det", params_det), ("ptt", params_ptt)]:
        streamlines = generate_tractogram(seeds,
                                          initial_directions,
                                          sc,
                                          params,
                                          pmf_gen)
        connectome = connectivity_matrix(streamlines, affine, labels)[1:,1:]

        r, _ = pearsonr(GT_connectome[connectome_mask].flatten(),
                        connectome[connectome_mask].flatten())

        npt.assert_(r > 0.9, msg="Algorithm " + algo + " has a low performance score: " + str(r))
