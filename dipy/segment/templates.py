from dipy.tracking.streamline import set_number_of_points
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.segment.clustering import QuickBundles
from dipy.align.streamlinear import whole_brain_slr
from dipy.tracking.utils import move_streamlines
import numpy as np


def bundle_to_wb(fixed_wb, moving_wb, moving_bundle):
    # TODO:implement a 2-step registration in which you align whole brain, do
    # recobundles, then SLR to that bundle to lock on to the local anatomy

    moved, transform, qb_cents1, qb_cents2 = whole_brain_slr(fixed_wb,
                                                             moving_wb,
                                                             verbose=True,
                                                             progressive=True)
    bundle_xfmd = moving_bundle.copy().apply_affine(transform).streamline
    # bundle_xfmd = nib.streamlines.tractogram.Tractogram(moving_bundle,
    #             affine_to_rasmm = fa_aff).streamlines
    return moved, transform, bundle_xfmd


def make_rb_template(bundle_list, keystone_boi, qb_thresh=5., Nsubsamp=20,
                     clsz_thresh=5, keystone2MNI_xfm=None, verbose=False):
    '''
    bundle_list: list of independent bundles (lists) not assumed to be in the
                 same space
    keystone_boi: bundle (list) of streamlines that will be the anchor bundle
                  all others are registered to for the template
    qb_thresh: threshold for quickbundle (determines how finely each bundle is
               clustered)
    Nsubsamp: subsampling for quickbundles and SLR
    clsz_thresh: how many streamlines a cluster must have to be included in the
                 template*
    keystone2MNI_SLR: streamlinear registration between the whole brain
                      keystone and MNI**
    verbose: print info about each bundle as it runs

    *qb_thresh adn clsz_thresh are related. If you have a fine parcellation
    (low qb_thresh) then the clsz_threshold should be quite low since clusters
    will be small.

    **PROVIDE THIS IF (and only if) YOU WANT THE RESULT TO BE IN MNI SPACE
    OTHERWISE IT WILL BE IN KEYSTONE SPACE (KEYSTONE PATIENT'S DIFFUSION SPACE)
    '''

    template_sls = []
    rejected_sls = []
    template_labels = []

    boi_sls_subsamp = set_number_of_points(keystone_boi, Nsubsamp)
    for i, sls in enumerate(bundle_list):
        print(len(bundle_list)-i)
        sls_subsamp = set_number_of_points(sls, Nsubsamp)
        qb = QuickBundles(threshold=qb_thresh)
        clusters = qb.cluster(sls)
        cluster_sizes = [len(cl) for cl in clusters]

        # enforce that clusters smaller than a threshold are not in template
        centroids = clusters.centroids
        slr = StreamlineLinearRegistration()
        srm = slr.optimize(static=boi_sls_subsamp, moving=sls_subsamp)
        xfmd_centroids = srm.transform(centroids)

        # TODO: we actually want to upsample the centroids so the template has
        # better properties... what's the most efficient way to do that?

        for j, b in enumerate(xfmd_centroids):
            if cluster_sizes[j] < clsz_thresh:
                rejected_sls.append(xfmd_centroids.pop(j))
        template_sls += xfmd_centroids
        template_labels += list(i * np.ones(len(xfmd_centroids), int))
        if verbose:
            print('Bundle %i' % i)
            print('N centroids: %i' % len(centroids))
            print('kept %i rejected %i total %i' % (len(template_sls),
                                                    len(rejected_sls),
                                                    len(clusters)))
    if keystone2MNI_xfm:
        # TODO: settle on how to implement MNI transform...
        # implement a 2-step registration in which you align whole brain, do
        # recobundles, then SLR to that bundle to lock on to the local anatomy

        print('Transforming to MNI space...')
        template_sls = list(move_streamlines(template_sls, keystone2MNI_xfm))
        rejected_sls = list(move_streamlines(rejected_sls, keystone2MNI_xfm))

    return template_sls, rejected_sls, template_labels
