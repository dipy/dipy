"""
===========================
Automatic bundle extraction
===========================
"""
import numpy as np
from os.path import basename
import nibabel as nib
import nibabel.trackvis as tv
from glob import glob
from dipy.viz import fvtk
from time import time, sleep

from dipy.tracking.streamline import (length,
                                      transform_streamlines,
                                      set_number_of_points,
                                      select_random_set_of_streamlines)
from dipy.segment.clustering import QuickBundles
from dipy.tracking.distances import (bundles_distances_mdf,
                                     bundles_distances_mam)
from itertools import chain, izip
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.viz.axycolor import distinguishable_colormap
from os import mkdir
from os.path import isdir
import os
from dipy.io.pickles import load_pickle, save_pickle

def read_trk(fname):
    streams, hdr = tv.read(fname, points_space='rasmm')
    return [i[0] for i in streams], hdr


def write_trk(fname, streamlines, hdr=None):
    streams = ((s, None, None) for s in streamlines)
    if hdr is not None:
        hdr_dict = {key: hdr[key] for key in hdr.dtype.names}
        #hdr2 = deepcopy(hdr)
        tv.write(fname, streams, hdr_mapping=hdr_dict, points_space='rasmm')
    else:
        tv.write(fname, streams, points_space='rasmm')


def show_bundles(static, moving, linewidth=1., tubes=False,
                 opacity=1., fname=None):

    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1.)

    if tubes:
        static_actor = fvtk.streamtube(static, fvtk.colors.red,
                                       linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.streamtube(moving, fvtk.colors.green,
                                       linewidth=linewidth, opacity=opacity)

    else:
        static_actor = fvtk.line(static, fvtk.colors.red,
                                 linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.line(moving, fvtk.colors.green,
                                 linewidth=linewidth, opacity=opacity)

    fvtk.add(ren, static_actor)
    fvtk.add(ren, moving_actor)

    fvtk.add(ren, fvtk.axes(scale=(2, 2, 2)))

    fvtk.show(ren, size=(900, 900))
    if fname is not None:
        fvtk.record(ren, size=(900, 900), out_path=fname)


def show_centroids(clusters, colormap=None, cam_pos=None,
                   cam_focal=None, cam_view=None,
                   magnification=1, fname=None, size=(900, 900)):

    bg = (1, 1, 1)
    if colormap is None:
        colormap = distinguishable_colormap(bg=bg)

    ren = fvtk.ren()
    ren.SetBackground(*bg)

    max_cz = np.max(map(len, clusters))
    for cluster, color in izip(clusters, colormap):
            fvtk.add(ren, fvtk.line(cluster.centroid,
                                    color, linewidth=len(cluster)*10./float(max_cz)))

    fvtk.show(ren, size=size)
    if fname is not None:
        fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                    out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                    magnification=magnification, size=size, verbose=True)


def get_bounding_box(streamlines):
    box_min = np.array([np.inf, np.inf, np.inf])
    box_max = -np.array([np.inf, np.inf, np.inf])

    for s in streamlines:
        box_min = np.minimum(box_min, np.min(s, axis=0))
        box_max = np.maximum(box_max, np.max(s, axis=0))

    return box_min, box_max


def show_clusters_grid_view(clusters, colormap=None, makelabel=None,
                            cam_pos=None, cam_focal=None, cam_view=None,
                            magnification=1, fname=None, size=(900, 900),
                            tubes=False):

    def grid_distribution(N):
        def middle_divisors(n):
            for i in range(int(n ** (0.5)), 2, -1):
                if n % i == 0:
                    return i, n // i

            return middle_divisors(n+1)  # If prime number take next one

        height, width = middle_divisors(N)
        X, Y, Z = np.meshgrid(np.arange(width), np.arange(height), [0])
        return np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    bg = (1, 1, 1)

    #if colormap is None:
    #    colormap = distinguishable_colormap(bg=bg)

    positions = grid_distribution(len(clusters))

    box_min, box_max = get_bounding_box(chain(*clusters))

    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)

    width, height, depth = box_max - box_min
    text_scale = [height*0.1] * 3
    cnt = 0
    for cluster, color, pos in izip(clusters, colormap, positions):
        offset = pos * (box_max - box_min)
        offset[0] += pos[0] * 4*text_scale[0]
        offset[1] += pos[1] * 4*text_scale[1]

        if tubes:
            fvtk.add(ren, fvtk.streamtube([s + offset for s in cluster],
                                          [color]*len(cluster)))
        else:
            fvtk.add(ren, fvtk.line([s + offset for s in cluster],
                                    [color]*len(cluster)))

        if makelabel is not None:
            #label = makelabel(cluster)
            label = makelabel[cnt]
            #text_scale = tuple([scale / 50.] * 3)
            text_pos = offset + np.array([0, height+4*text_scale[1], depth])/2.
            text_pos[0] -= len(label) / 2. * text_scale[0]


            fvtk.label(ren, text=label, pos=text_pos, scale=text_scale,
                       color=(0, 0, 0))

        cnt += 1
    fvtk.show(ren, size=size)


def remove_clusters_by_size(clusters, min_size=0):
    #sizes = np.array(map(len, clusters))
    #mean_size = sizes.mean()
    #std_size = sizes.std()

    by_size = lambda c: len(c) >= min_size
    #and len(c) >= mean_size - alpha * std_size

    # filter returns a list of clusters
    return filter(by_size, clusters)


def whole_brain_registration(streamlines1, streamlines2,
                             rm_small_clusters=50,
                             maxiter=100,
                             select_random=None,
                             verbose=False):

    if verbose:
        print(len(streamlines1))
        print(len(streamlines2))

    def check_range(streamline, gt=50, lt=250):

        if (length(streamline) > gt) & (length(streamline) < lt):
            return True
        else:
            return False

    streamlines1 = [s for s in streamlines1 if check_range(s)]
    streamlines2 = [s for s in streamlines2 if check_range(s)]

    if verbose:
        print(len(streamlines1))
        print(len(streamlines2))

    if select_random is not None:
        rstreamlines1 = select_random_set_of_streamlines(streamlines1,
                                                         select_random)
    else:
        rstreamlines1 = streamlines1

    rstreamlines1 = set_number_of_points(rstreamlines1, 20)
    qb1 = QuickBundles(threshold=15)
    rstreamlines1 = [s.astype('f4') for s in rstreamlines1]
    cluster_map1 = qb1.cluster(rstreamlines1)
    clusters1 = remove_clusters_by_size(cluster_map1, rm_small_clusters)
    qb_centroids1 = [cluster.centroid for cluster in clusters1]

    if select_random is not None:
        rstreamlines2 = select_random_set_of_streamlines(streamlines2,
                                                         select_random)
    else:
        rstreamlines2 = streamlines2

    rstreamlines2 = set_number_of_points(rstreamlines2, 20)
    qb2 = QuickBundles(threshold=15)
    rstreamlines2 = [s.astype('f4') for s in rstreamlines2]
    cluster_map2 = qb2.cluster(rstreamlines2)
    clusters2 = remove_clusters_by_size(cluster_map2, rm_small_clusters)
    qb_centroids2 = [cluster.centroid for cluster in clusters2]

    slr = StreamlineLinearRegistration(x0='affine',
                                       options={'maxiter': maxiter})

    t = time()

    slm = slr.optimize(qb_centroids1, qb_centroids2)

    if verbose:
        print('QB1 %d' % len(qb_centroids1,))
        print('QB2 %d' % len(qb_centroids2,))

    duration = time() - t
    if verbose:
        print('SAR done in  %f seconds.' % (duration, ))

    print('SAR iterations: %d ' % (slm.iterations, ))

    moved_streamlines2 = slm.transform(streamlines2)

    return moved_streamlines2, slm.matrix, qb_centroids1, qb_centroids2


def janice_next_subject(dname_whole_streamlines, verbose=False):

    for wb_trk2 in glob(dname_whole_streamlines + '*.trk'):

        wb2, hdr = read_trk(wb_trk2)

        if verbose:
            print(wb_trk2)

        tag = basename(wb_trk2).split('_')[0]

        if verbose:
            print(tag)

        yield (wb2, tag)


def janice_manual(tag, bundle_type, dname_model_bundles):
    trk = dname_model_bundles + tag + '/tracts/' + \
        bundle_type + '/' + tag + '_' + bundle_type + '_GP.trk'
    print('Reading ' + trk)
    streamlines, _ = read_trk(trk)

    return streamlines


def janice_initial(model_tag='t0337', bundle_type='IFOF_R'):

    initial_dir = '/home/eleftherios/Data/Hackethon_bdx/'

    dname_model_bundles = initial_dir + 'bordeaux_tracts_and_stems/'

    model_bundle_trk = dname_model_bundles + \
        model_tag + '/tracts/' + bundle_type + '/' + \
        model_tag + '_' + bundle_type + '_GP.trk'

    model_bundle, _ = read_trk(model_bundle_trk)

    dname_whole_brain = initial_dir + \
        'bordeaux_whole_brain_DTI/whole_brain_trks_60sj/'

    if model_tag in ['t0336', 't0337', 't0340', 't0364']:
        model_streamlines_trk = dname_whole_brain + \
            model_tag + '_dti_mean02_fact-45_splined.trk'
    else:
        model_streamlines_trk = dname_whole_brain + \
            model_tag + '_dti_mean02_fact_45.trk'

    model_streamlines, hdr = read_trk(model_streamlines_trk)

    results_dir = initial_dir + 'results_' + \
        model_tag + '_' + bundle_type + '/'

    if not isdir(results_dir):
        mkdir(results_dir)

    ret = initial_dir, results_dir, dname_model_bundles, \
        dname_whole_brain, model_bundle, model_streamlines, hdr

    return ret


def bundle_adjacency(dtracks0, dtracks1, threshold):
    # d01 = distance_matrix(MinimumAverageDirectFlipMetric(),
    #                       dtracks0, dtracks1)
    d01 = bundles_distances_mdf(dtracks0, dtracks1)

    pair12 = []
    solo1 = []

    for i in range(len(dtracks0)):
        if np.min(d01[i, :]) < threshold:
            j = np.argmin(d01[i, :])
            pair12.append((i, j))
        else:
            solo1.append(dtracks0[i])

    pair12 = np.array(pair12)
    pair21 = []

    solo2 = []
    for i in range(len(dtracks1)):
        if np.min(d01[:, i]) < threshold:
            j = np.argmin(d01[:, i])
            pair21.append((i, j))
        else:
            solo2.append(dtracks1[i])

    pair21 = np.array(pair21)
    A = len(pair12) / np.float(len(dtracks0))
    B = len(pair21) / np.float(len(dtracks1))
    res = 0.5 * (A + B)
    return res


def auto_extract(model_bundle, moved_streamlines,
                 close_centroids_thr=20,
                 clean_thr=7.,
                 local_slr=True,
                 disp=False, verbose=True, expand_thr=None):

    if verbose:
        print('# Centroids of model bundle')

    t0 = time()

    rmodel_bundle = set_number_of_points(model_bundle, 20)
    rmodel_bundle = [s.astype('f4') for s in rmodel_bundle]

    qb = QuickBundles(threshold=20)
    model_cluster_map = qb.cluster(rmodel_bundle)
    model_centroids = model_cluster_map.centroids

    if verbose:
        print('Duration %f ' % (time() - t0, ))

    if verbose:
        print('# Calculate centroids of moved_streamlines')

    t = time()

    rstreamlines = set_number_of_points(moved_streamlines, 20)
    # qb.cluster had problem with f8
    rstreamlines = [s.astype('f4') for s in rstreamlines]

    cluster_map = qb.cluster(rstreamlines)
    cluster_map.refdata = moved_streamlines

    if verbose:
        print('Duration %f ' % (time() - t, ))

    if verbose:
        print('# Find centroids which are close to the model_centroids')

    t = time()

    centroid_matrix = bundles_distances_mdf(model_centroids,
                                            cluster_map.centroids)

    centroid_matrix[centroid_matrix > close_centroids_thr] = np.inf

    mins = np.min(centroid_matrix, axis=0)
    close_clusters = [cluster_map[i] for i in np.where(mins != np.inf)[0]]

    #close_centroids = [cluster.centroid for cluster in close_clusters]

    close_streamlines = list(chain(*close_clusters))

    if verbose:
        print('Duration %f ' % (time() - t, ))

    if disp:
        show_bundles(model_bundle, close_streamlines)

    if local_slr:

        if verbose:
            print('# Local SLR of close_streamlines to model')

        t = time()

        x0 = np.array([0, 0, 0, 0, 0, 0, 1.])
        bounds = [(-30, 30), (-30, 30), (-30, 30),
                  (-45, 45), (-45, 45), (-45, 45), (0.5, 1.5)]

        slr = StreamlineLinearRegistration(x0=x0, bounds=bounds)

        static = select_random_set_of_streamlines(model_bundle, 400)
        moving = select_random_set_of_streamlines(close_streamlines, 600)

        static = set_number_of_points(static, 20)
        #static = [s.astype('f4') for s in static]
        moving = set_number_of_points(moving, 20)
        #moving = [m.astype('f4') for m in moving]

        slm = slr.optimize(static, moving)

        closer_streamlines = transform_streamlines(close_streamlines,
                                                   slm.matrix)

        if verbose:
            print('Duration %f ' % (time() - t, ))

        if disp:
            show_bundles(model_bundle, closer_streamlines)

        matrix = slm.matrix
    else:
        closer_streamlines = close_streamlines
        matrix = np.eye(4)

    if verbose:
        print('# Remove streamlines which are a bit far')

    t = time()

    rcloser_streamlines = set_number_of_points(closer_streamlines, 20)

    clean_matrix = bundles_distances_mdf(rmodel_bundle, rcloser_streamlines)

    clean_matrix[clean_matrix > clean_thr] = np.inf

    mins = np.min(clean_matrix, axis=0)
    close_clusters_clean = [closer_streamlines[i]
                            for i in np.where(mins != np.inf)[0]]

    if verbose:
        print('Duration %f ' % (time() - t, ))

    msg = 'Total duration of automatic extraction %0.4f seconds.'
    print(msg % (time() - t0, ))
    if disp:
        show_bundles(model_bundle, close_clusters_clean)

    if expand_thr is not None:
        rclose_clusters_clean = set_number_of_points(close_clusters_clean, 20)
        expand_matrix = bundles_distances_mam(rclose_clusters_clean,
                                              rcloser_streamlines)

        expand_matrix[expand_matrix > expand_thr] = np.inf
        mins = np.min(expand_matrix, axis=0)
        expanded = [closer_streamlines[i]
                    for i in np.where(mins != np.inf)[0]]

        return expanded, matrix

    return close_clusters_clean, matrix


def exp_validation_with_janice(model_tag='t0337',
                               bundle_type='IFOF_R',
                               close_centroids_thr=20,
                               clean_thr=5.,
                               local_slr=True,
                               verbose=True,
                               disp=False,
                               expand_thr=None,
                               f_extracted=None,
                               f_manual=None,
                               f_model=None):

    # Read Janice's model streamlines
    model_tag = model_tag  # 't0253'
    bundle_type = bundle_type  # 'IFOF_R'
    print(model_tag)

    ret = janice_initial(model_tag, bundle_type)

    initial_dir, results_dir, model_bundles_dir, whole_brain_dir, \
        model_bundle, model_streamlines, hdr = ret

    list_of_all = []
    list_of_m_vs_e = []
    list_of_manual = []

    bas = []

    i = 0

    for (streamlines, tag) in janice_next_subject(whole_brain_dir):

        print(tag)
        print('# Affine registration')
        t = time()

        ret = whole_brain_registration(model_streamlines, streamlines,
                                       maxiter=150)
        moved_streamlines, mat, centroids1, centroids2 = ret

        print('Duration %f ' % (time() - t, ))

        #write_trk(tag + '_moved_streamlines.trk', moved_streamlines, hdr=hdr)

        #show_bundles(centroids1, centroids2)
        #show_bundles(centroids1, transform_streamlines(centroids2, mat))

        extracted, mat2 = auto_extract(model_bundle, moved_streamlines,
                                       close_centroids_thr=close_centroids_thr,
                                       clean_thr=clean_thr,
                                       local_slr=local_slr,
                                       disp=disp, verbose=verbose,
                                       expand_thr=expand_thr)

        result_trk = results_dir + tag + '_extracted.trk'

        print('Writing ' + result_trk)

        write_trk(result_trk, extracted, hdr=hdr)

        manual = janice_manual(tag, bundle_type, model_bundles_dir)

        manual_in_model = transform_streamlines(manual,
                                                np.dot(mat2, mat))

        if disp:
            show_bundles(manual_in_model, extracted)

        list_of_all.append(extracted)

        list_of_m_vs_e.append(extracted)
        list_of_m_vs_e.append(manual_in_model)

        list_of_manual.append(manual_in_model)

        ba = bundle_adjacency(set_number_of_points(manual_in_model),
                              set_number_of_points(extracted), 0.5)
        bas.append(ba)

        print ('BA : %f ' % (ba, ))
        print

        i += 1
        if i == 5:
            break

    list_of_all.append(model_bundle)

    colormap = np.random.rand(len(list_of_all), 3)
    colormap[-1] = np.array([1., 0, 0])

    show_clusters_grid_view(list_of_all, colormap)

    colormap2 = np.random.rand(len(list_of_m_vs_e), 3)
    show_clusters_grid_view(list_of_m_vs_e, colormap2)

    if f_extracted is not None:

        save_pickle(f_extracted, list_of_all[:-1])
        save_pickle(f_manual, list_of_manual)
        save_pickle(f_model, model_bundle)

    return bas


def exp_fancy_data(model_tag='Renauld',
                   bundle_type='cst.right',
                   close_centroids_thr=20,
                   clean_thr=5.,
                   local_slr=True,
                   verbose=True,
                   disp=False):


    dname = '/home/eleftherios/Data/fancy_data/'
    #2013_07_15_Alexandra/TRK_files/bundles_cst.right.trk'
    for model_dir in glob(dname + '*' + model_tag):
        model_streamlines, hdr = read_trk(model_dir + '/streamlines_500K.trk')
        model_bundle, hdr = read_trk(model_dir + '/TRK_files/bundles_' + bundle_type + '.trk')

    group = ['Girard'] #, 'Vanier', 'Delattre', 'Aubin', 'Butler2']
    #team_B = ['Renauld', 'St-Jean', 'Owji', 'Castonguay', 'Marcil']

    print(model_tag)

    list_of_all = []
    list_of_all_labels = []
    list_of_m_vs_e = []
    list_of_m_vs_e_labels = []

    for subj in group:

        print(subj)

        for subj_dir in glob(dname + '*' + subj):

            streamlines, hdr = read_trk(subj_dir + '/streamlines_500K.trk')
            manual_bundle, hdr = read_trk(subj_dir + '/TRK_files/bundles_' + bundle_type + '.trk')

            ret = whole_brain_registration(model_streamlines, streamlines,
                                           maxiter=150, select_random=50000, verbose=verbose)
            moved_streamlines, mat, centroids1, centroids2 = ret

            # print('Duration %f ' % (time() - t, ))

            extracted, mat2 = auto_extract(model_bundle, moved_streamlines,
                                           close_centroids_thr=close_centroids_thr,
                                           clean_thr=clean_thr,
                                           local_slr=local_slr,
                                           disp=disp, verbose=verbose)

            #show_bundles(model_bundle, extracted)

            manual_bundle_in_model = transform_streamlines(manual_bundle, np.dot(mat2, mat))
            #show_bundles(manual_bundle_in_model, extracted)

            list_of_all.append(extracted)
            list_of_all_labels.append('E')

            list_of_m_vs_e.append(extracted)
            list_of_m_vs_e_labels.append('E')

            list_of_m_vs_e.append(manual_bundle_in_model)
            list_of_m_vs_e_labels.append('MA')

    list_of_all.append(model_bundle)
    list_of_all_labels.append('M')

    colormap = np.random.rand(len(list_of_all), 3)
    colormap[-1] = np.array([1., 0, 0])

    show_clusters_grid_view(list_of_all, colormap, list_of_all_labels)

    colormap2 = np.random.rand(len(list_of_m_vs_e), 3)
    show_clusters_grid_view(list_of_m_vs_e, colormap2, list_of_m_vs_e_labels)

    save_pickle(bundle_type + 'list_of_all.pkl', list_of_all)
    save_pickle(bundle_type + 'list_of_m_vs_e.pkl', list_of_m_vs_e)
    save_pickle(bundle_type + 'list_of_all_labels.pkl', list_of_all_labels)
    save_pickle(bundle_type + 'list_of_m_vs_e_labels.pkl', list_of_m_vs_e_labels)


def show_fancy_data_results(bundle_type):

    list_of_all = load_pickle(bundle_type + 'list_of_all.pkl')
    list_of_m_vs_e = load_pickle(bundle_type + 'list_of_m_vs_e.pkl', )
    list_of_all_labels = load_pickle(bundle_type + 'list_of_all_labels.pkl')
    list_of_m_vs_e_labels = load_pickle(bundle_type + 'list_of_m_vs_e_labels.pkl')

    colormap = np.random.rand(len(list_of_all), 3)
    colormap[-1] = np.array([1., 0, 0])

    show_clusters_grid_view(list_of_all, colormap, list_of_all_labels)

    colormap2 = np.random.rand(len(list_of_m_vs_e), 3)
    show_clusters_grid_view(list_of_m_vs_e, colormap2, list_of_m_vs_e_labels)


def get_camilles_bundles(bundle_type='all'):

    dname = '/home/eleftherios/bundle_paper/data/faisceaux/'

    ffa = '/home/eleftherios/Data/MPI_Elef/fa_1x1x1.nii.gz'

    affine = nib.load(ffa).get_affine()

    bundle_names = ['CC_front', 'CC_middle', 'CC_back', \
                    'cingulum_left', 'cingulum_right', \
                    'CST_left', 'CST_right', \
                    'IFO_left', 'IFO_right', \
                    'ILF_left', 'ILF_right',
                    'SCP_left', 'SCP_right', \
                    'SLF_left', 'SLF_right', \
                    'uncinate_left', 'uncinate_right']

    if bundle_type == 'all':

        streamlines = []
        for b in bundle_names:
            streams, hdr = tv.read(dname + b + '.trk')
            bundle = [b[0] for b in streams]
            bundle = transform_streamlines(bundle, affine)
            streamlines += bundle

        return streamlines

    else:
        streams, hdr = tv.read(dname + bundle_type + '.trk')
        bundle = [b[0] for b in streams]
        bundle = transform_streamlines(bundle, affine)
        return bundle


def exp_tumor_data(model_tag,
                   bundle_type,
                   close_centroids_thr=20,
                   clean_thr=5.,
                   local_slr=True,
                   verbose=True,
                   disp=False,
                   expand_thr=None,
                   fname=None):

    #model_streamlines = get_camilles_bundles('all')
    #model_bundle = get_camilles_bundles('CST_left')

    dname = '/home/eleftherios/Data/fancy_data/'
    for model_dir in glob(dname + '*' + model_tag):
        model_streamlines, hdr = read_trk(model_dir + '/streamlines_500K.trk')
        model_bundle, hdr = read_trk(model_dir + '/TRK_files/bundles_' + bundle_type + '.trk')


    dname = '/home/eleftherios/Data/National_Geographic/forElef/'
    streamlines, _ = read_trk(dname + 'whole_brain_fa_0.01_mask.trk')

    manual_bundle_left, _ = read_trk(dname + 'CST_tumor_max.trk')
    manual_bundle_right, _ = read_trk(dname + 'CST_healthy_max.trk')

    # show_bundles(model_streamlines[:5000], streamlines[:5000])

    ret = whole_brain_registration(model_streamlines, streamlines,
                                   maxiter=150, select_random=50000,
                                   verbose=verbose)
    moved_streamlines, mat, centroids1, centroids2 = ret

    # show_bundles(model_streamlines[:5000], moved_streamlines[:5000])

    # print('Duration %f ' % (time() - t, ))

    extracted, mat2 = auto_extract(model_bundle, moved_streamlines,
                                   close_centroids_thr=close_centroids_thr,
                                   clean_thr=clean_thr,
                                   local_slr=local_slr,
                                   disp=disp, verbose=verbose,
                                   expand_thr=expand_thr)

    # show_bundles(model_bundle, extracted)
    colormap = np.array([[1, 0, 0.], [0, 1, 0.], [0, 0, 1.]])

    if bundle_type == 'cst.left':
        manual_bundle = manual_bundle_left
    if bundle_type == 'cst.right':
        manual_bundle = manual_bundle_right

    manual_bundle_in_model = transform_streamlines(manual_bundle,
                                                   np.dot(mat2, mat))

    m_e_ma = [model_bundle, extracted, manual_bundle_in_model]

    if verbose:
        show_clusters_grid_view(m_e_ma,
                                colormap)

    if fname is not None:
        save_pickle(fname, m_e_ma)


def load_m_e_ma(fname):

    m_e_ma = load_pickle(fname)
    colormap = np.array([[1, 0, 0.], [0, 1, 0.], [0, 0, 1.]])
    show_clusters_grid_view(m_e_ma, colormap)


def load_m_e_ma_one(fname):

    m_e_ma = load_pickle(fname)

    colormap = np.array([[1, 0, 0.], [0, 1, 0.], [0, 0, 1.]])

    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1.)

    model = m_e_ma[0]

    model_actor = fvtk.line(model, fvtk.colors.red, linewidth=3)
    fvtk.add(ren, model_actor)
    fvtk.show(ren, size=(1200, 1200))
    sleep(1)
    fvtk.record(ren, size=(1200, 1200), magnification=1, out_path='model_cst.png')
    model_actor.GetProperty().SetOpacity(0)


    extracted_actor = fvtk.line(m_e_ma[1], fvtk.colors.orange, linewidth=3)
    fvtk.add(ren, extracted_actor)
    fvtk.show(ren,size=(1200, 1200))
    sleep(1)
    fvtk.record(ren, size=(1200, 1200), magnification=1,
                out_path='extracted_cst.png')
    fvtk.rm(ren, extracted_actor)

    extracted_actor = fvtk.line(m_e_ma[2], fvtk.colors.orange, linewidth=3)
    fvtk.add(ren, extracted_actor)
    fvtk.show(ren,size=(1200, 1200))
    sleep(1)
    fvtk.record(ren, size=(1200, 1200), magnification=1,
                out_path='manual_cst.png')
    fvtk.rm(ren, extracted_actor)


def load_janice_results(f_extracted, f_manual, f_model):

    np.random.seed(20)

    extracted = load_pickle(f_extracted)

    manual = load_pickle(f_manual)

    model = load_pickle(f_model)

    colormap = np.random.rand(len(extracted) + 1, 3)

    colormap[-1] = np.array([1, 0, 0.])

    show_clusters_grid_view(extracted + [model], colormap, size=(1200, 1200),
                            tubes=True)

    show_clusters_grid_view(manual + [model], colormap, size=(1200, 1200),
                            tubes=True)

    #show_clusters_grid_view([model] + [model] + )

    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1.)

    model_actor = fvtk.line(model, fvtk.colors.red, linewidth=3)
    fvtk.add(ren, model_actor)
    fvtk.show(ren, size=(1200, 1200))
    sleep(1)
    fvtk.record(ren, size=(1200, 1200), magnification=1, out_path='0.png')
    model_actor.GetProperty().SetOpacity(0)

    from dipy.viz.fvtk import colors as c

    colors = [c.blue, c.green, c.orange, c.black, c.magenta]

    for i in range(len(extracted)):
        extracted_actor = fvtk.line(extracted[i],colors[i], linewidth=3)
        fvtk.add(ren, extracted_actor)
        fvtk.show(ren,size=(1200, 1200))
        sleep(1)
        fvtk.record(ren, size=(1200, 1200), magnification=1,
                    out_path=str(i) + '_extracted.png')
        fvtk.rm(ren, extracted_actor)

        manual_actor = fvtk.line(manual[i], colors[i], linewidth=3)
        fvtk.add(ren, manual_actor)
        fvtk.show(ren,size=(1200, 1200))
        sleep(1)
        fvtk.record(ren, size=(1200, 1200), magnification=1,
                    out_path=str(i) + '_manual.png')
        fvtk.rm(ren, manual_actor)




if __name__ == '__main__':

    bas = exp_validation_with_janice(model_tag='t0337',
                                     bundle_type='UNC_R',
                                     close_centroids_thr=20,
                                     clean_thr=5.,
                                     local_slr=True,
                                     verbose=True,
                                     disp=False,
                                     expand_thr=2.,
                                     f_extracted='extracted.pkl',
                                     f_manual='manual.pkl',
                                     f_model='model.pkl')
    plot(bas, 'o')

    load_janice_results(f_extracted='extracted.pkl',
                        f_manual='manual.pkl',
                        f_model='model.pkl')

#    bas = exp_fancy_data(model_tag='Renauld',
#                         bundle_type='cst.right',
#                         close_centroids_thr=20,
#                         clean_thr=5.,
#                         local_slr=True,
#                         verbose=True,
#                         disp=False)

#    dname_res = '/home/eleftherios/postdoc/ismrm2015/tumor_results/'
#    exp_tumor_data(model_tag='Girard',
#                   bundle_type='cst.left',
#                   close_centroids_thr=20,
#                   clean_thr=5.,
#                   local_slr=True,
#                   verbose=True,
#                   disp=False,
#                   expand_thr=None,
#                   fname = dname_res + 'Girard_cst.left_expand_None.pkl')
#
#
#    exp_tumor_data(model_tag='Girard',
#                   bundle_type='cst.left',
#                   close_centroids_thr=20,
#                   clean_thr=5.,
#                   local_slr=True,
#                   verbose=True,
#                   disp=False,
#                   expand_thr=5.,
#                   fname = dname_res + 'Girard_cst.left_expand_5.pkl')
#
#    exp_tumor_data(model_tag='Girard',
#                   bundle_type='cst.right',
#                   close_centroids_thr=20,
#                   clean_thr=5.,
#                   local_slr=True,
#                   verbose=True,
#                   disp=False,
#                   expand_thr=None,
#                   fname = dname_res + 'Girard_cst.right_expand_None.pkl')
#
#
#    exp_tumor_data(model_tag='Girard',
#                   bundle_type='cst.right',
#                   close_centroids_thr=20,
#                   clean_thr=5.,
#                   local_slr=True,
#                   verbose=True,
#                   disp=False,
#                   expand_thr=5.,
#                   fname = dname_res + 'Girard_cst.right_expand_5.pkl')
