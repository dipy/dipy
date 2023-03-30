import matplotlib.pyplot as plt
from pycpd import DeformableRegistration
import numpy as np
from scipy.optimize import linear_sum_assignment
from dipy.align.streamlinear import slr_with_qbx
from dipy.tracking.streamline import (unlist_streamlines,
                                      Streamlines)
from dipy.stats.analysis import assignment_map

from dipy.segment.metricspeed import MinimumAverageDirectFlipMetric

from dipy.segment.metricspeed import dist



def mdf(s1, s2):

    return dist(MinimumAverageDirectFlipMetric(), s1, s2)

def mdf_dist(cb1, cb2):

    n = len(cb1)
    m = len(cb2)
    dist = np.zeros((m,n))

    for i in range(m):
        s1 = cb2[i]
        for j in range(n):
            s2 = cb1[j]
            dist[i][j] = mdf(s1,s2)

    return dist

def find_missing(lst, cb):
    return [x for x in range(0, len(cb))
                               if x not in lst]


def bundlewarp(static, moving, dist=None, alpha=0.3, beta=20, max_iter=15, affine=True, precomputed=False):

    if affine:
        moving_aligned, _, _, _ = slr_with_qbx(static, moving, rm_small_clusters=0)

    else:
        # rigid
        moving_aligned, _, _, _ = slr_with_qbx(static, moving, x0='rigid', rm_small_clusters=0)

    if precomputed==True:
        print("using pre-computed distances")
    else:
        dist = mdf_dist(static, moving_aligned)


    matched_pairs = np.zeros((len(moving), 2))
    matched_pairs1 = np.asarray(linear_sum_assignment(dist)).T

    for mt in matched_pairs1:
            matched_pairs[mt[0]] = mt

    num=len(matched_pairs1)

    all_pairs = list(matched_pairs1[:,0])
    all_matched = False

    while all_matched==False:

        num = len(all_pairs)

        if num < len(moving):

            ml = find_missing(all_pairs, moving)

            dist2 = dist[:][ml] #saves computation



            #index changes
            # dist2 has distance among unmatched streamlines of moving bundle and all static bundle's streamlines
            matched_pairs2 = np.asarray(linear_sum_assignment(dist2)).T

            for i in range(matched_pairs2.shape[0]):
                matched_pairs2[i][0] = ml[matched_pairs2[i][0]]

            for mt in matched_pairs2:
                matched_pairs[mt[0]] = mt


            all_pairs.extend(matched_pairs2[:,0])

            num2 = num+len(matched_pairs2)
            if num2==len(moving):
                all_matched=True
                num = num2
        else:
            all_matched=True

    deformed_bundle = Streamlines([])
    warp = []
    # iterate over each pair of streamlines and deform them
    # append in deformed_bundle
    for i in range(len(matched_pairs)):

        pairs = matched_pairs[i]
        s1 = static[int(pairs[1])]
        s2 = moving_aligned[int(pairs[0])]

        static_s = s1
        moving_s = s2

        reg = DeformableRegistration(**{'X': static_s, 'Y': moving_s, 'alpha': alpha, 'beta': beta, 'max_iterations':max_iter})
        ty, pr = reg.register()
        deformed_bundle.append(ty)
        warp.append(pr)

    # returns affinely moved bundle, deformed bundle, streamline correspondences, and warp field
    return moving_aligned, deformed_bundle, dist, matched_pairs, warp

def bundlewarp_vector_filed(moving_aligned, deformed_bundle):
    """Vector field computation as the difference between each streamline point in
    the deformed and aligned bundles"""
    points_aligned, _ = unlist_streamlines(moving_aligned)
    points_deformed, _ = unlist_streamlines(deformed_bundle)
    vector_field = points_deformed - points_aligned

    offsets = np.sqrt(np.sum((vector_field)**2, 1))  # vector field modules

    "Normalize vectors to be unitary (directions)"
    directions = vector_field / np.array([offsets]).T

    """"Define colors mapping the direction vectors to RGB.
    Absolute value generates DTI-like colors"""
    colors = directions

    return offsets, directions, colors


def bundlewarp_shape_analysis(moving_aligned, deformed_bundle, no_disks=10):

    n = no_disks
    offsets, directions, colors = bundlewarp_vector_filed(moving_aligned, deformed_bundle)

    indx = assignment_map(deformed_bundle, deformed_bundle, n)
    indx = np.array(indx)

    colors = [np.random.rand(3) for si in range(n)]

    disks_color = []
    for i in range(len(indx)):
        disks_color.append(tuple(colors[indx[i]]))

    x = np.array(range(1,n+1))
    shape_profile = np.zeros(n)
    std_1 = np.zeros(n)
    std_2 = np.zeros(n)

    for i in range(n):

        shape_profile[i]=np.mean(offsets[indx==i])
        stdv=np.std(offsets[indx==i])
        std_1[i]=shape_profile[i]+stdv
        std_2[i]=shape_profile[i]-stdv

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(x, shape_profile, '-', label='Mean', color='Purple', linewidth=3, markersize=12)
    ax.fill_between(x, std_1, std_2, alpha=0.2, label='Std', color='Purple',)

    plt.xticks(x)
    plt.ylim(0,max(std_1)+2)

    plt.ylabel("Average Displacement")
    plt.xlabel("Segment Number")
    plt.title('Bundle Shape Profile')
    plt.legend(loc=2)