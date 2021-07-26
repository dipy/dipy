"""Atlasing module: utilities to compute population specific bundle atlases.

Available functions:
    get_pairwise_tree: computes the matching pairs for a given number of items.

    combine_bundle: combines two bundles into a single one using different
    streamline combination methods.

    compute_bundle_atlas: given a list of input bundles computes the population
    atlas.
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from time import sleep
from dipy.align.bundlemin import distance_matrix_mdf
from dipy.align.streamlinear import slr_with_qbx
from dipy.data import fetch_bundle_atlas_hcp842, get_bundle_atlas_hcp842
from dipy.io.streamline import (create_tractogram_header, load_tractogram,
                                save_trk, Space, StatefulTractogram)
from dipy.tracking.streamline import (orient_by_streamline, relist_streamlines,
                                      select_random_set_of_streamlines,
                                      set_number_of_points, Streamlines,
                                      unlist_streamlines)
from dipy.viz import actor, window


def get_pairwise_tree(n_item, seed=None):
    """Pairwise tree structure calculation.

    Constructs a pairwise tree by randomly matching the indexes of a given
    number of items. The computed index structure is intended to be used for
    atlasing, where the items (e.g. bundles) are combined in pairs until a
    single item (atlas) is obtained.

    Parameters
    ----------
    n_item : int
        Number of items to be matched.
    seed : float, optional
        Seed for reproducibility. Default is None.

    Returns
    -------
    matched_pairs : list of array
        Each element in list is an array of (N x 2) with the indexes of the
        N items to be matched in a certain level.
    alone : list of array
        Index of the items not combined at a certain level when the number of
        items to be matched is odd. When it is even this value is set to None.
    n_reg : list of int
        Number of pairwise registrations at each level of the tree.
    """
    if type(n_item) != int:
        raise TypeError("n_item must be an integer > 1")
    if n_item <= 1:
        raise ValueError("n_item must be > 1")

    np.random.seed(seed)

    matched_pairs = []
    alone = []
    n_reg = []

    while n_item > 1:
        # Define indexes
        index = np.arange(n_item)
        # Compute the number of registration pairs
        n_reg.append(np.floor(n_item/2).astype('int'))
        # Shuffle the bundles
        index = np.random.permutation(n_item)
        # If n_bundles is odd duplicate one of the others
        if np.mod(n_item, 2) == 1:
            # avoid removing again an item twice (index == 0)
            lonely = np.random.randint(1, n_item)
            index = np.delete(index, np.where(index == lonely))
            alone.append(lonely)
        else:
            alone.append(None)
        # Generate pairwise registration matrix
        index = np.reshape(index, (n_reg[-1], 2))
        # Update bundle number
        n_item = np.ceil(n_item/2).astype('int')
        # Save matched pairs
        matched_pairs.append(index)

    return matched_pairs, alone, n_reg


def combine_bundles(bundle1, bundle2, comb_method='rlap', distance='mdf',
                    n_stream=2000):
    """Combine two bundles.

    Combines two bundles into a single one by using different methods to match,
    average and pick the streamlines from the two bundles. Bundles need to be
    already in the same space and streamlines must have the same number of
    points.

    Parameters
    ----------
    bundle1 : list
        Streamline coordinates as a list of 2D ndarrays of shape[-1]==3
    bundle2 : list
        Streamline coordinates as a list of 2D ndarrays of shape[-1]==3
    comb_method : str, optional
        Method to be used to combine the two bundles. Default is 'rlap_keep'.
    distance : str, optional
        Distance used for streamline matching. Used by all methods except for
        'random_pick'.  Default is 'mdf'.
    n_stream : int, optional
        Number of streamlines to be selected when comb_method='random_pick'.
        Default is 2000.

    Returns
    -------
    combined : list
        Streamline coordinates of the combined bundle as a list of 2D ndarrays
        of shape[-1]==3.
    """
    # If random_pick just merge all streamlines and pick n_stream randomly
    if comb_method == 'random_pick':
        bundles = np.concatenate((bundle1, bundle2))
        return select_random_set_of_streamlines(bundles, n_stream)

    def distance_matrix_mdf_start_end(bundle_1, bundle_2):
        bundle_1 = set_number_of_points(bundle_1, 2)
        bundle_2 = set_number_of_points(bundle_2, 2)
        return distance_matrix_mdf(bundle_1, bundle_2)

    if distance == 'mdf':
        distance = distance_matrix_mdf
    elif distance == 'mdf_se':
        distance = distance_matrix_mdf_start_end
    else:
        raise ValueError("Incorrect distance metric")

    # Set as bundle 1 the one with less streamlines
    if len(bundle2) < len(bundle1):
        aux = bundle1.copy()
        bundle1 = bundle2
        bundle2 = aux

    # Compute distance matrix
    cost = distance(bundle1, bundle2)

    combined = []

    if comb_method == 'rlap':
        # Minimize the sum of distances (RLAP)
        matched_pairs = np.asarray(linear_sum_assignment(cost)).T

        for ind1, ind2 in matched_pairs:

            stream1 = bundle1[ind1]
            stream2 = bundle2[ind2]

            stream2 = orient_by_streamline([stream2], stream1)
            stream2, _ = unlist_streamlines(stream2)

            stream_mean = np.mean([stream1, stream2], axis=0)

            combined.append(stream_mean)

    elif comb_method == 'rlap_closest':

        n_stream = len(bundle2)

        # Solve the linear assignment problem
        ind_lap1, ind_lap2 = linear_sum_assignment(cost)

        for ind2 in range(n_stream):
            # Check if streamline already matched by LAP
            aux = np.argwhere(ind_lap2 == ind2)
            if aux.size > 0:
                ind1 = ind_lap1[aux[0][0]]
            else:
                # Find the closest streamline and save it
                ind1 = np.argmin(cost[:, ind2])

            # Get matched streamline pair
            stream1 = bundle1[ind1]
            stream2 = bundle2[ind2]

            # Reorient streamlines in the right order
            stream2 = orient_by_streamline([stream2], stream1)
            stream2, _ = unlist_streamlines(stream2)

            # Combine matched streamlines and generate the atlas
            stream_mean = np.mean([stream1, stream2], axis=0)

            # Store streamline
            combined.append(stream_mean)

    elif comb_method == 'rlap_keep':

        n_stream = len(bundle2)

        # Solve the linear assignment problem
        ind_lap1, ind_lap2 = linear_sum_assignment(cost)

        for ind2 in range(n_stream):
            # Check if streamline already matched by LAP
            aux = np.argwhere(ind_lap2 == ind2)
            # If matched compute mean
            if aux.size > 0:
                ind1 = ind_lap1[aux[0][0]]
                # Get matched streamline pair
                stream1 = bundle1[ind1]
                stream2 = bundle2[ind2]
                # Reorient streamlines in the right order
                stream2 = orient_by_streamline([stream2], stream1)
                stream2, _ = unlist_streamlines(stream2)

                # Combine matched streamlines and generate the atlas
                combined.append(np.mean([stream1, stream2], axis=0))

            # If not matched keep it as it is
            else:
                combined.append(bundle2[ind2])
    else:
        raise ValueError("Not supported bundle combination method")
    return combined


def show_bundles(bundles, fname, colors=None):
    """Show and/or save bundle renderization.

    Renders the the input bundles and saves them to the fname file.

    Parameters
    ----------
    bundles : list
        Bundles to be rendered.
    fname : str
        File to save renderization image.
    colors : list, optional
        Colors of each bundle. If None, then random colors are used.

    Returns
    -------
    None.

    """
    n_bundle = len(bundles)

    if colors is None:
        colors = list(np.random.rand(n_bundle, 3))

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)
    scene.set_camera((0, 0, 300), (12, 18, 40), (0, 1, 0))
    for i, bundle in enumerate(bundles):

        lines_actor = actor.streamtube(bundle, colors[i],
                                       opacity=0.5, linewidth=0.1)

        # Guess the optimal visualization angle based on coordinates
        coords, _ = unlist_streamlines(bundle)
        in_left = len(np.where(coords[:, 0] < 0)[0])/coords.shape[0]
        in_right = len(np.where(coords[:, 0] > 0)[0])/coords.shape[0]
        if abs(in_left-in_right) < 0.2:
            rY, rZ = 0, 0
        elif in_left > in_right:
            rY, rZ = 90, 90
        else:
            rY, rZ = -90, -90
        lines_actor.RotateZ(rZ)
        lines_actor.RotateY(rY)

        scene.add(lines_actor)
    sleep(1)  # necessary?
    window.record(scene, n_frames=1, out_path=fname, size=(900, 900))


def compute_atlas_bundle(in_dir, subjects=None, group=None, mid_path='',
                         bundle_names=None, model_bundle_dir=None,
                         out_dir=None, merge_out=False, save_temp=False,
                         n_stream_min=10, n_stream_max=5000, n_point=20,
                         distance='mdf_se', comb_method='rlap',
                         skip_pairs=False):
    """Compute a population specific bundle atlas.

    Given several segmented bundles as input, compute the atlas by combining
    the bundles pairwise.

    Parameters
    ----------
    in_dir : str
        Input folder.
    subjects : str, optional
        Path to a BIDS-like participants.tsv file with the IDs of the subjects
        to be processed. If None, all folders in ``in_dir`` are considered as
        subjects.
    group : str, optional
        Label to select a subject group when the tsv file defining subjects
        has a ``group`` column. If None, all subjects are processed.
    mid_path : str, optional
        Intermediate path between ``in_dir`` and bundle files. Default is ''.
    bundle_names : str, optional
        Path to a tsv file with the names of the bundles to be processed. If
        None, all trk files of the first subject will be considered as
        bundle_names.
    model_bundle_dir : str, optional
        Directory with reference bundles to be used as a reference to move all
        bundles to a standard space. If 'HCP-842' the HCP-842 atlas bundles are
        downloaded and used. If None, bundles are assumed to be in the same
        space and no registration is performed.
    out_dir : str, optional
        Output directory. If None, the current directory is used.
    merge_out : boolean, optional
        If True the resulting atlases of all bundles are combined into a single
        file. Default is False.
    save_temp : boolean, optional
        If True the intermediate results of each tree level are saved in a temp
        folder. Default is False.
    n_stream_min : int, optional
        Bundles with less than ``n_stream_min`` streamlines wont be processed.
        Default is 10.
    n_stream_max : int, optional
        Bundles with more than ``n_stream_max`` streamlines are cropped to have
        that number and speed up the computation. Default is 5000.
    n_point : int, optional
        All streamlines are set to have ``n_point`` points. Default is 20.
    distance : str, optional
        Distance metric to be used to combine bundles. Default is 'mdf_se'.
    comb_method : str, optional
        Method to be used to combine each bundle pair. Default is 'rlap'.
    skip_pairs : boolean, optional
        If true bundle combination steps are randomly skipped. This helps to
        obtain a sharper result. Default is False.

    Returns
    -------
    atlas : list of Streamlines
        A list with the computed atlas bundles.
    atlas_merged : Streamlines
        A single bundle containing all the computed atlas bundles together.
    """
    if type(in_dir) != str:
        raise TypeError("in_dir must be a string")
    if type(mid_path) != str:
        raise TypeError("mid_path must be a string")
    if type(n_point) != int:
        raise TypeError("n_point must be an int")
    if os.path.isdir(in_dir) is False:
        raise ValueError("Input directory does not exist")
    if out_dir is None:
        out_dir = os.getcwd()
    if os.path.isdir(out_dir) is False:
        raise ValueError("Output directory does not exist")
    if n_stream_min < 1:
        raise ValueError("n_stream_min must be >= 1")
    if n_stream_max < 1:
        raise ValueError("n_stream_max must be >= 1")
    if n_point < 2:
        raise ValueError("n_point must be >= 2")

    print('Input directory:' + in_dir)
    print('Output directory:' + out_dir)

    # Get subjects
    if subjects is None:
        files = os.listdir(in_dir)
        subjects = [file for file in files if os.path.isdir(
                    os.path.join(in_dir, file))]
    else:
        # Read participants.tsv file (BIDS format)
        df = pd.read_csv(subjects, delimiter='\t', dtype='object')
        if group is None:
            subjects = list(df.participant)
        else:
            subjects = list(df.loc[df.group == group].participant)
    subjects.sort()  # necessary?
    if len(set(subjects)) < len(subjects):
        raise ValueError("Subjects cannot be duplicated")
    print(str(len(subjects)) + " subjects to be processed:")
    print(subjects)

    # Get bundle names
    if bundle_names is None:
        bundle_dir = os.path.join(in_dir, subjects[0], mid_path)
        print("Retrieving bundle names from " + bundle_dir)
        if os.path.isdir(bundle_dir) is False:
            raise ValueError("Path to subject bundles is incorrect")

        files = os.listdir(bundle_dir)
        my_files = [file for file in files if file.endswith('.trk')]
        bundle_names = [os.path.splitext(file)[0] for file in my_files]
    else:
        # Read bundles.tsv file (first column)
        df = pd.read_csv(bundle_names, delimiter='\t', dtype='object')
        bundle_names = list(df.iloc[:, 0])
    bundle_names.sort()  # necessary?
    if len(set(bundle_names)) < len(bundle_names):
        raise ValueError("Bundle names cannot be duplicated")
    print(str(len(bundle_names)) + " bundles to be processed:")
    print(bundle_names)

    # Get bundle file list
    bundle_files = {}
    for sub in subjects:
        for bundle in bundle_names:
            file = os.path.join(in_dir, sub, mid_path, bundle + '.trk')
            bundle_files[(sub, bundle)] = file

    # Get model bundle list
    if model_bundle_dir is None:
        model_bundles = None
    else:
        if model_bundle_dir == 'HCP-842':
            _, _ = fetch_bundle_atlas_hcp842()
            _, all_bundles_files = get_bundle_atlas_hcp842()
            model_bundle_dir = os.path.split(all_bundles_files)[0]

        files = os.listdir(model_bundle_dir)
        my_files = [file for file in files if file.endswith('.trk')]
        model_bundle_list = [os.path.splitext(file)[0] for file in my_files]

        if not all(x in model_bundle_list for x in bundle_names):
            raise ValueError("Not all the specified bundles have a model")
        model_bundles = {}
        for bundle in bundle_names:
            model_bundles[bundle] = os.path.join(
                model_bundle_dir, bundle + '.trk')

    # Create temporary folder
    temp_dir = os.path.join(out_dir, 'temp/')
    if os.path.exists(temp_dir):
        print("There is already a temp folder in out_dir. Deleting.")
        shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)

    # Atlas building starts
    atlas = []
    for bundle in bundle_names:
        print("Processing bundle: " + bundle)

        step_dir = os.path.join(temp_dir, bundle, 'step_0')
        os.makedirs(step_dir)

        # Load model bundle if required
        if model_bundles is not None:
            file = model_bundles[bundle]
            bundle_obj = load_tractogram(file, reference='same',
                                         bbox_valid_check=False)
            model_bundle = bundle_obj.streamlines
            header_model \
                = create_tractogram_header(file, *bundle_obj.space_attributes)

        # Preprocess all bundles
        file_list = []
        for i, sub in enumerate(subjects):
            file = bundle_files[(sub, bundle)]
            bundle_obj = load_tractogram(file, reference='same',
                                         bbox_valid_check=False)
            streamlines = bundle_obj.streamlines
            header = create_tractogram_header(file,
                                              *bundle_obj.space_attributes)

            n_stream = len(streamlines)
            if n_stream < n_stream_min:
                print(f"{file} has {n_stream} streamlines. Discarded.")
                continue
            elif n_stream > n_stream_max:
                streamlines = select_random_set_of_streamlines(
                    streamlines, n_stream_max)

            streamlines = set_number_of_points(streamlines, n_point)

            if model_bundles is not None:
                streamlines, _, _, _ = slr_with_qbx(static=model_bundle,
                                                    moving=streamlines,
                                                    x0='affine',
                                                    rm_small_clusters=1,
                                                    qbx_thr=[5])
                header = header_model

            file = f'{step_dir}/bundle_{i}_prev_{sub}'
            file_list.append(file)
            new_tractogram = StatefulTractogram(streamlines, reference=header,
                                                space=Space.RASMM)
            save_trk(new_tractogram, f'{file}.trk', bbox_valid_check=False)
            show_bundles([streamlines], f'{file}.png')

        print("Bundle preprocessing: ok.")

        # Compute pairwise registration tree-structure
        tree, alone, n_reg = get_pairwise_tree(n_item=len(file_list))

        # Go through all tree levels
        for i_step, pairs in enumerate(tree):
            new_file_list = list()

            # Create step folder
            new_step_dir = os.path.join(
                temp_dir, bundle, 'step_' + str(i_step+1))
            os.mkdir(new_step_dir)

            # A lonely bundle goes to the next level
            has_lonely = 0
            if alone[i_step] is not None:
                has_lonely = 1

                file_new = f'{new_step_dir}/bundle_0_prev_{alone[i_step]}'
                new_file_list.append(file_new)
                shutil.copyfile(f'{file_list[alone[i_step]]}.trk',
                                f'{file_new}.trk')

            # Loop through all registration pairs
            for index, pair in enumerate(pairs):
                i = pair[0]
                j = pair[1]

                print(f"step:{i_step+1}/{len(tree)}" +
                      f" pair:{index+1}/{n_reg[i_step]}")

                file = file_list[i] + '.trk'
                bundle_obj = load_tractogram(file, reference='same',
                                             bbox_valid_check=False)
                static = bundle_obj.streamlines
                header = create_tractogram_header(file,
                                                  *bundle_obj.space_attributes)

                file = file_list[j] + '.trk'
                moving = load_tractogram(file, reference='same',
                                         bbox_valid_check=False).streamlines

                aligned, _, _, _ = slr_with_qbx(static, moving, x0='affine',
                                                rm_small_clusters=1,
                                                qbx_thr=[5])

                points, offsets = unlist_streamlines(static)
                static = relist_streamlines(points, offsets)
                points, offsets = unlist_streamlines(aligned)
                aligned = relist_streamlines(points, offsets)
                points, offsets = unlist_streamlines(moving)
                moving = relist_streamlines(points, offsets)

                # Combine bundles
                if skip_pairs and np.random.choice([True, False], 1)[0]:
                    combined = combine_bundles(static, aligned, 'random_pick',
                                               distance)
                else:
                    combined = combine_bundles(static, aligned, comb_method,
                                               distance)

                file = f'{new_step_dir}/bundle_{index+has_lonely}_prev_{i}_{j}'
                new_file_list.append(file)
                new_tractogram = StatefulTractogram(Streamlines(combined),
                                                    reference=header,
                                                    space=Space.RASMM)
                save_trk(new_tractogram, file + ".trk", bbox_valid_check=False)
                show_bundles([static, moving, aligned], f'{file}_reg.png',
                             colors=[(0, 0, 1), (1, 0, 0), (0, 1, 0)])
                show_bundles([Streamlines(combined)], f'{file}.png')

            step_dir = new_step_dir
            file_list = new_file_list

        shutil.copyfile(file + '.trk', out_dir + '/' + bundle + '.trk')

        atlas.append(Streamlines(combined))

    if not save_temp:
        shutil.rmtree(temp_dir)

    if merge_out:
        atlas_merged = np.concatenate(atlas)
        file = f'{out_dir}/whole_brain.trk'
        new_tractogram = StatefulTractogram(atlas_merged, reference=header,
                                            space=Space.RASMM)
        save_trk(new_tractogram, file, bbox_valid_check=False)
        return atlas, atlas_merged
    return atlas
