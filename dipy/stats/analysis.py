from __future__ import division, print_function, absolute_import

import os
import numpy as np
from time import time
from scipy import spatial
from scipy.spatial import cKDTree
from scipy.ndimage.interpolation import map_coordinates
from scipy.spatial.distance import mahalanobis

from dipy.utils.optpkg import optional_package
from dipy.io.image import load_nifti
from dipy.io.streamline import load_trk
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.io.peaks import load_peaks
from dipy.tracking.streamline import (set_number_of_points,
                                      values_from_volume,
                                      orient_by_streamline,
                                      transform_streamlines,
                                      Streamlines)
from glob import glob
pd, have_pd, _ = optional_package("pandas")
_, have_tables, _ = optional_package("tables")

if have_pd:
    import pandas as pd


def _save_hdf5(fname, dt, col_name, col_size=16):
    """ Saves the given input dataframe to .h5 file

    Parameters
    ----------
    fname : string
        file name for saving the hdf5 file
    dt : Pandas DataFrame
        DataFrame to be saved as .h5 file
    col_name : string
        column name to have specific column size
    col_size : integer
        min column size (default=5)

    """

    df = pd.DataFrame(dt)
    filename_hdf5 = fname + '.h5'

    store = pd.HDFStore(filename_hdf5, complevel=9)
    store.append(fname, df, data_columns=True, complevel=9)#,
                 #min_itemsize={col_name: col_size})
    store.close()


def peak_values(bundle, peaks, dt, pname, bname, subject, group, ind, dir):
    """ Peak_values function finds the peak direction and peak value of a point
        on a streamline used while tracking (generating the tractogram) and
        save it in hd5 file.

        Parameters
        ----------
        bundle : string
            Name of bundle being analyzed
        peaks : peaks
            contains peak directions and values
        dt : DataFrame
            DataFrame to be populated
        pname : string
            Name of the dti metric
        bname : string
            Name of bundle being analyzed.
        subject : string
            subject number as a string (e.g. 10001)
        group : string
            which group subject belongs to (e.g. patient or control)
        ind : integer list
            ind tells which disk number a point belong.
        dir : string
            path of output directory

    """


    gfa = peaks.gfa
    dti_measures(bundle, gfa, dt, pname+'_gfa', bname, subject, group, ind, dir)

    qa = peaks.qa[...,0]
    dti_measures(bundle, qa, dt, pname+'_qa', bname, subject, group, ind, dir)


def dti_measures(bundle, metric, dt, pname, bname, subject, group, ind, dir):
    """ Calculates dti measure (eg: FA, MD) per point on streamlines and
        save it in hd5 file.

        Parameters
        ----------
        bundle : string
            Name of bundle being analyzed
        metric : matrix of float values
            dti metric e.g. FA, MD
        dt : DataFrame
            DataFrame to be populated
        pname : string
            Name of the dti metric
        bname : string
            Name of bundle being analyzed.
        subject : string
            subject number as a string (e.g. 10001)
        group : string
            which group subject belongs to (e.g. patient or control)
        ind : integer list
            ind tells which disk number a point belong.
        dir : string
            path of output directory
    """

    if group == 'patient':
        group_id = 1 # 1 means patient
    else:
        group_id = 0 # 0 means control

    #dt["bundle"] = []
    dt["streamline"] = []
    dt["disk#"] = []
    dt["subject"] = []
    dt[pname] = []
    dt["group"] = []

    values = map_coordinates(metric, bundle._data.T,
                             order=1)

    dt["disk#"].extend(ind[list(range(len(values)))]+1)
    #dt["bundle"].extend([bname]*len(values))
    dt["subject"].extend([subject]*len(values))
    dt["group"].extend([group_id]*len(values))
    dt[pname].extend(values)
    
    
    for st_i in range(len(bundle)):
    
        st = bundle[st_i]
        dt["streamline"].extend([st_i]*len(st))
        
    

    #devide and conquer
    file_name = bname+"_"+pname


    _save_hdf5(os.path.join(dir, file_name), dt, col_name="bundle")

def assignment_map(target_bundle, model_bundle, no_disks):

    mbundle_streamlines = set_number_of_points(model_bundle,
                                           nb_points=no_disks)

    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=85., metric=metric)
    clusters = qb.cluster(mbundle_streamlines)
    centroids = Streamlines(clusters.centroids)

    _, indx = cKDTree(centroids.data, 1,
                      copy_data=True).query(target_bundle.data, k=1)
    
    return indx
    
def bundle_analysis(model_bundle_folder, bundle_folder, orig_bundle_folder,
                    metric_folder, group, subject, no_disks=100,
                    out_dir=''):
    """
    Applies statistical analysis on bundles and saves the results
    in a directory specified by ``out_dir``.

    Parameters
    ----------
    model_bundle_folder : string
        Path to the input model bundle files. This path may contain
        wildcards to process multiple inputs at once.
    bundle_folder : string
        Path to the input bundle files in common space. This path may
        contain wildcards to process multiple inputs at once.
    orig_folder : string
        Path to the input bundle files in native space. This path may
        contain wildcards to process multiple inputs at once.
    metric_folder : string
        Path to the input dti metric or/and peak files. It will be used as
        metric for statistical analysis of bundles.
    group : string
        what group subject belongs to e.g. control or patient
    subject : string
        subject id e.g. 10001
    no_disks : integer, optional
        Number of disks used for dividing bundle into disks. (Default 100)
    out_dir : string, optional
        Output directory (default input file directory)

    References
    ----------
    .. [Chandio19] Chandio, B.Q., S. Koudoro, D. Reagan, J. Harezlak,
    E. Garyfallidis, Bundle Analytics: a computational and statistical
    analyses framework for tractometric studies, Proceedings of:
    International Society of Magnetic Resonance in Medicine (ISMRM),
    Montreal, Canada, 2019.

    """


    t = time()

    dt = dict()

    #mb = os.listdir(model_bundle_folder)
    
    #mb = glob(model_bundle_folder+"/*.trk")
    # edit here remove this later, uncomment above line
    mb = glob(os.path.join(model_bundle_folder,"*.trk"))
    print(mb)
    #edit end
    
    mb.sort()
    #bd = os.listdir(bundle_folder)
    #bd.sort()
    
    #bd = glob(bundle_folder+"/*.trk")
    bd = glob(os.path.join(bundle_folder,"*.trk"))
    
    
    bd.sort()
    print(bd)
    
    #org_bd = os.listdir(orig_bundle_folder)
    org_bd = glob(os.path.join(orig_bundle_folder,"*.trk"))
    org_bd.sort()
    print(org_bd)
    n = len(org_bd)
    n = len(mb)

    for io in range(n):
        #mbundles = load_trk(mb[io], reference='same', bbox_valid_check=False) #os.path.join(model_bundle_folder, mb[io]))
       # mbundles = mbundles.streamlines
        
        
        mbundles, _ = load_trk(mb[io])
        
       # bundles = load_trk(bd[io], reference='same', bbox_valid_check=False) #os.path.join(bundle_folder, bd[io]))
        #bundles = bundles.streamlines
        
        
        bundles, _ = load_trk(bd[io])
        
        #orig_bundles = load_trk(org_bd[io], reference='same', bbox_valid_check=False)#os.path.join(orig_bundle_folder,
                                   #org_bd[io]))
        #orig_bundles = orig_bundles.streamlines


        orig_bundles, _ = load_trk(org_bd[io])

        if len(orig_bundles) > 5 :

            '''
            mbundle_streamlines = set_number_of_points(mbundles,
                                                       nb_points=no_disks)
    
            metric = AveragePointwiseEuclideanMetric()
            qb = QuickBundles(threshold=85., metric=metric)
            clusters = qb.cluster(mbundle_streamlines)
            centroids = Streamlines(clusters.centroids)
    
            print('Number of centroids ', len(centroids.data))
            print('Model bundle ', mb[io])
            print('Number of streamlines in bundle in common space ',
                  len(bundles))
            print('Number of streamlines in bundle in original space ',
                  len(orig_bundles))
    
            _, indx = cKDTree(centroids.data, 1,
                              copy_data=True).query(bundles.data, k=1)
    
            '''
            indx = assignment_map(bundles, mbundles, no_disks)
            ind = np.array(indx)
            #metric_files_names = os.listdir(metric_folder)
            metric_files_names_dti = glob(os.path.join(metric_folder,"*.nii.gz"))
            
            metric_files_names_csa = glob(os.path.join(metric_folder,"*.pam5"))
            #metric_files_names = ["csd_peaks.pam5"]
            #["csa_peaks.pam5", "csd_peaks.pam5"] 
            #["fa.nii.gz", "md.nii.gz", "ad.nii.gz", "rd.nii.gz"]
            
            
            _, affine = load_nifti(metric_files_names_dti[0])
    
            affine_r = np.linalg.inv(affine)
            transformed_orig_bundles = transform_streamlines(orig_bundles,
                                                             affine_r)
    
            for mn in range(len(metric_files_names_dti)):
    
                ab = os.path.split(metric_files_names_dti[mn])
                metric_name = ab[1]
                
                fm = metric_name[:-7]
                bm = os.path.split(mb[io])[1][:-4]
                #bm = mb[io][14:-4]
                print("bm = ", bm)

                dt = dict()
                #metric_name = os.path.join(metric_folder,
                #                           metric_files_names_dti[mn])
                
                
                
    
                print("metric = ", metric_files_names_dti[mn])

                metric, _ = load_nifti(metric_files_names_dti[mn])

                dti_measures(transformed_orig_bundles, metric, dt, fm,
                             bm, subject, group, ind, out_dir)


            for mn in range(len(metric_files_names_csa)):
                ab = os.path.split(metric_files_names_csa[mn])
                metric_name = ab[1]
                
                fm = metric_name[:-7]
                bm = os.path.split(mb[io])[1][:-4]
                #bm = mb[io][14:-4]
                print("bm = ", bm)
                print("metric = ", metric_files_names_csa[mn])
                dt = dict()
                metric = load_peaks(metric_files_names_csa[mn])
                #peak_values(bundles, metric, dt, fm, bm, subject, group,
                #            ind, out_dir)
                peak_values(transformed_orig_bundles, metric, dt, fm, bm,
                            subject, group, ind, out_dir)



    print("total time taken in minutes = ", (-t + time())/60)


def gaussian_weights(bundle, n_points=100, return_mahalnobis=False,
                     stat=np.mean):
    """
    Calculate weights for each streamline/node in a bundle, based on a
    Mahalanobis distance from the core the bundle, at that node (mean, per
    default).

    Parameters
    ----------
    bundle : Streamlines
        The streamlines to weight.
    n_points : int, optional
        The number of points to resample to. *If the `bundle` is an array, this
        input is ignored*. Default: 100.

    Returns
    -------
    w : array of shape (n_streamlines, n_points)
        Weights for each node in each streamline, calculated as its relative
        inverse of the Mahalanobis distance, relative to the distribution of
        coordinates at that node position across streamlines.
    """
    # Resample to same length for each streamline:
    bundle = set_number_of_points(bundle, n_points)

    # This is the output
    w = np.zeros((len(bundle), n_points))

    # If there's only one fiber here, it gets the entire weighting:
    if len(bundle) == 1:
        if return_mahalnobis:
            return np.array([np.nan])
        else:
            return np.array([1])

    for node in range(n_points):
        # This should come back as a 3D covariance matrix with the spatial
        # variance covariance of this node across the different streamlines
        # This is a 3-by-3 array:
        node_coords = bundle.data[node::n_points]
        c = np.cov(node_coords.T, ddof=0)
        # Reorganize as an upper diagonal matrix for expected Mahalnobis input:
        c = np.array([[c[0, 0], c[0, 1], c[0, 2]],
                      [0, c[1, 1], c[1, 2]],
                      [0, 0, c[2, 2]]])
        # Calculate the mean or median of this node as well
        # delta = node_coords - np.mean(node_coords, 0)
        m = stat(node_coords, 0)
        # Weights are the inverse of the Mahalanobis distance
        for fn in range(len(bundle)):
            # In the special case where all the streamlines have the exact same
            # coordinate in this node, the covariance matrix is all zeros, so
            # we can't calculate the Mahalnobis distance, we will instead give
            # each streamline an identical weight, equal to the number of
            # streamlines:
            if np.allclose(c, 0):
                w[:, node] = len(bundle)
                break
            # Otherwise, go ahead and calculate Mahalanobis for node on
            # fiber[fn]:
            w[fn, node] = mahalanobis(node_coords[fn], m, np.linalg.inv(c))
    if return_mahalnobis:
        return w
    # weighting is inverse to the distance (the further you are, the less you
    # should be weighted)
    w = 1 / w
    # Normalize before returning, so that the weights in each node sum to 1:
    return w / np.sum(w, 0)


def afq_profile(data, bundle, affine=None, n_points=100,
                orient_by=None, weights=None, **weights_kwarg):
    """
    Calculates a summarized profile of data for a bundle or tract
    along its length.

    Follows the approach outlined in [Yeatman2012]_.

    Parameters
    ----------
    data : 3D volume
        The statistic to sample with the streamlines.

    bundle : StreamLines class instance
        The collection of streamlines (possibly already resampled into an array
         for each to have the same length) with which we are resampling. See
         Note below about orienting the streamlines.

    affine: 4-by-4 array, optional.
        A transformation associated with the streamlines in the bundle.
        Default: identity.

    n_points: int, optional
        The number of points to sample along the bundle. Default: 100.

    orient_by: streamline, optional.
        A streamline to use as a standard to orient all of the streamlines in
        the bundle according to.

    weights : 1D array or 2D array or callable (optional)
        Weight each streamline (1D) or each node (2D) when calculating the
        tract-profiles. Must sum to 1 across streamlines (in each node if
        relevant). If callable, this is a function that calculates weights.

    weights_kwarg : key-word arguments
        Additional key-word arguments to pass to the weight-calculating
        function. Only to be used if weights is a callable.

    Returns
    -------
    ndarray : a 1D array with the profile of `data` along the length of
        `bundle`

    Note
    ----
    Before providing a bundle as input to this function, you will need to make
    sure that the streamlines in the bundle are all oriented in the same
    orientation relative to the bundle (use :func:`orient_by_streamline`).

    References
    ----------
    .. [Yeatman2012] Yeatman, Jason D., Robert F. Dougherty,
       Nathaniel J. Myall, Brian A. Wandell, and Heidi M. Feldman. 2012.
       "Tract Profiles of White Matter Properties: Automating Fiber-Tract
       Quantification" PloS One 7 (11): e49790.
    """
    if orient_by is not None:
        bundle = orient_by_streamline(bundle, orient_by, affine=affine)
    if len(bundle) == 0:
        raise ValueError("The bundle contains no streamlines")

    # Resample each streamline to the same number of points:
    fgarray = set_number_of_points(bundle, n_points)

    # Extract the values
    values = np.array(values_from_volume(data, fgarray, affine=affine))

    if weights is None:
        weights = np.ones(values.shape) / values.shape[0]
    elif callable(weights):
        weights = weights(bundle, **weights_kwarg)
    else:
        # We check that weights *always sum to 1 across streamlines*:
        if not np.allclose(np.sum(weights, 0), np.ones(n_points)):
            raise ValueError("The sum of weights across streamlines must ",
                             "be equal to 1")

    return np.sum(weights * values, 0)
