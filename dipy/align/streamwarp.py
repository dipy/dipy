import numpy as np
from scipy.optimize import linear_sum_assignment
from dipy.align.streamlinear import slr_with_qbx
from dipy.tracking.streamline import (unlist_streamlines,
                                      Streamlines)
from dipy.stats.analysis import assignment_map
from dipy.utils.optpkg import optional_package

from dipy.align.bundlemin import distance_matrix_mdf

pycpd, have_pycpd, _ = optional_package("pycpd")
matplt, have_matplotlib, _ = optional_package("matplotlib")

if have_pycpd:
    from pycpd import DeformableRegistration

if have_matplotlib:
    import matplotlib.pyplot as plt


def find_missing(lst, cb):
    """Find unmatched streamline indices in moving bundle.

    Parameters
    ----------
    lst : int
        List containing all the streamlines indices in moving bundle.
    cb : int
        List containing streamline indices of the moving bundle that were not
        matched to any streamline in static bundle.

    Returns
    -------
    list
        List containing unmatched streamlines from moving bundle

    """
    return [x for x in range(0, len(cb)) if x not in lst]


def bundlewarp(static, moving, dist=None, alpha=0.3, beta=20, max_iter=15,
               affine=True, precomputed=False):
    """Register two bundle using nonlinear method.

    Parameters
    ----------
    static : Streamlines
        Reference/fixed bundle

    moving : Streamlines
        Target bundle that will be moved/registered to match the static bundle

    dist : float, optional.
        Precomputed distance matrix (default None)

    alpha : float, optional
        Represents the trade-off between regularizing the deformation and
        having points match very closely. Lower value of alpha means high
        deformations (default 0.3)

    beta : int, optional
        Represents the strength of the interaction between points
        Gaussian kernel size (default 20)

    affine : boolean, optional
        If False, use rigid registration as starting point (default True)

    References
    ----------
    .. [Chandio2023] Chandio et al. "BundleWarp, streamline-based nonlinear
            registration of white matter tracts." bioRxiv (2023): 2023-01.
    """
    x0 = 'affine' if affine else 'rigid'
    moving_aligned, _, _, _ = slr_with_qbx(static, moving, x0=x0,
                                           rm_small_clusters=0)

    if dist:
        print("using pre-computed distances")
    else:
        dist = distance_matrix_mdf(static, moving_aligned)

    matched_pairs = np.zeros((len(moving), 2))
    matched_pairs1 = np.asarray(linear_sum_assignment(dist)).T

    for mt in matched_pairs1:
        matched_pairs[mt[0]] = mt

    num = len(matched_pairs1)

    all_pairs = list(matched_pairs1[:, 0])
    all_matched = False

    while all_matched is False:

        num = len(all_pairs)

        if num < len(moving):

            ml = find_missing(all_pairs, moving)

            dist2 = dist[:][ml]

            # dist2 has distance among unmatched streamlines of moving bundle
            # and all static bundle's streamlines

            matched_pairs2 = np.asarray(linear_sum_assignment(dist2)).T

            for i in range(matched_pairs2.shape[0]):
                matched_pairs2[i][0] = ml[matched_pairs2[i][0]]

            for mt in matched_pairs2:
                matched_pairs[mt[0]] = mt

            all_pairs.extend(matched_pairs2[:, 0])

            num2 = num + len(matched_pairs2)
            if num2 == len(moving):
                all_matched = True
                num = num2
        else:
            all_matched = True

    deformed_bundle = Streamlines([])
    warp = []

    # Iterate over each pair of streamlines and deform them
    # Append deformed streamlines in deformed_bundle

    for pairs in enumerate(matched_pairs):

        pairs = matched_pairs[i]
        s1 = static[int(pairs[1])]
        s2 = moving_aligned[int(pairs[0])]

        static_s = s1
        moving_s = s2

        reg = DeformableRegistration(**{'X': static_s, 'Y': moving_s,
                                        'alpha': alpha, 'beta': beta,
                                        'max_iterations': max_iter})
        ty, pr = reg.register()
        deformed_bundle.append(ty)
        warp.append(pr)

    # Returns deformed bundle, affinely moved bundle, distance matrix,
    # streamline correspondences, and warp field
    return deformed_bundle, moving_aligned, dist, matched_pairs, warp


def bundlewarp_vector_filed(moving_aligned, deformed_bundle):
    """Calculate vector fields.

    Vector field computation as the difference between each streamline point
    in the deformed and linearly aligned bundles

    Parameters
    ----------
    moving_aligned : Streamlines
        Linearly (affinely) moved bundle
    deformed_bundle : Streamlines
        Nonlinearly (warped) bundle

    Returns
    -------
    offsets :
        vector field modules
    directions :
    colors :
    """
    points_aligned, _ = unlist_streamlines(moving_aligned)
    points_deformed, _ = unlist_streamlines(deformed_bundle)
    vector_field = points_deformed - points_aligned

    offsets = np.sqrt(np.sum((vector_field)**2, 1))  # vector field modules

    # Normalize vectors to be unitary (directions)
    directions = vector_field / np.array([offsets]).T

    # Define colors mapping the direction vectors to RGB.
    # Absolute value generates DTI-like colors
    colors = directions

    return offsets, directions, colors


def bundlewarp_shape_analysis(moving_aligned, deformed_bundle, no_disks=10):
    """Calculate bundle shape difference profile.

    Bundle shape difference analysis using magnitude from BundleWarp
    displacements and BUAN

    Parameters
    ----------
    moving_aligned : Streamlines
        Linearly (affinely) moved bundle
    deformed_bundle : Streamlines
        Nonlinearly (warped) bundle
    no_disks : Integer
        Number of segments to be created along the length of the bundle
        (Default 10)
    """
    n = no_disks
    offsets, directions, colors = bundlewarp_vector_filed(moving_aligned,
                                                          deformed_bundle)

    indx = assignment_map(deformed_bundle, deformed_bundle, n)
    indx = np.array(indx)

    colors = [np.random.rand(3) for si in range(n)]

    disks_color = []
    for ind in enumerate(indx):

        disks_color.append(tuple(colors[ind]))

    x = np.array(range(1, n+1))
    shape_profile = np.zeros(n)
    std_1 = np.zeros(n)
    std_2 = np.zeros(n)

    for i in range(n):

        shape_profile[i] = np.mean(offsets[indx == i])
        stdv = np.std(offsets[indx == i])
        std_1[i] = shape_profile[i]+stdv
        std_2[i] = shape_profile[i]-stdv

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(x, shape_profile, '-', label='Mean', color='Purple', linewidth=3,
            markersize=12)
    ax.fill_between(x, std_1, std_2, alpha=0.2, label='Std', color='Purple')

    plt.xticks(x)
    plt.ylim(0, max(std_1)+2)

    plt.ylabel("Average Displacement")
    plt.xlabel("Segment Number")
    plt.title('Bundle Shape Profile')
    plt.legend(loc=2)
