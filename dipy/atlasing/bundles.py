"""Atlasing module: utilities to compute population specific bundle atlases.

Available functions:
    get_pairwise_tree: computes the matching pairs for a given number of items.

    combine_bundle: combines two bundles into a single one using different
    streamline combination methods.

    compute_bundle_atlas: given a list of input bundles computes the population
    atlas.
"""
import logging
from os import listdir, mkdir, makedirs, getcwd
from os.path import join, isdir, splitext
from shutil import rmtree, copyfile
import numpy as np
from scipy.optimize import linear_sum_assignment
from dipy.align.bundlemin import distance_matrix_mdf
from dipy.align.streamlinear import slr_with_qbx
from dipy.io.streamline import (create_tractogram_header, load_tractogram,
                                save_trk, Space, StatefulTractogram)
from dipy.segment.clustering import QuickBundles
from dipy.tracking.streamline import (orient_by_streamline, relist_streamlines,
                                      select_random_set_of_streamlines,
                                      set_number_of_points, Streamlines,
                                      unlist_streamlines)
from dipy.utils.optpkg import optional_package

pd, has_pd, _ = optional_package('pandas')
_, has_fury, _ = optional_package('fury')

if has_fury:
    from dipy.io.utils import show_bundles

logger = logging.getLogger(__name__)


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
        raise TypeError("You provided a n_item input {n_item} of type \
                        {type(n_item)} but it must be an integer > 1")
    if n_item <= 1:
        raise ValueError("You provided a n_item input {n_item} that is not > \
                         1")

    np.random.seed(seed)

    matched_pairs = []
    alone = []
    n_reg = []

    while n_item > 1:
        # Define indexes
        index = np.arange(n_item)
        # Compute the number of registration pairs
        n_reg.append(np.floor(n_item/2).astype('int'))
        # Shuffle the items
        index = np.random.permutation(n_item)
        # If n_item is odd duplicate one of the others
        if np.mod(n_item, 2) == 1:
            # avoid removing again an item twice (index == 0)
            lonely = np.random.randint(1, n_item)
            index = np.delete(index, np.where(index == lonely))
            alone.append(lonely)
        else:
            alone.append(None)
        # Generate pairwise registration matrix
        index = np.reshape(index, (n_reg[-1], 2))
        # Update item number
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
        'random_pick'.  Default is 'mdf'. The 'mdf_se' distance uses only the
        start and end points.
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

    # Set as bundle 1 the bundle with less streamlines
    if len(bundle2) < len(bundle1):
        aux = bundle1.copy()
        bundle1 = bundle2
        bundle2 = aux

    # Reorient all streamlines at once based on centroid
    bundle1 = orient_by_streamline(bundle1, bundle1[0])
    qb = QuickBundles(threshold=50.)
    centroid = qb.cluster(bundle1).centroids[0]

    bundle1 = orient_by_streamline(bundle1, centroid)
    bundle2 = orient_by_streamline(bundle2, centroid)

    # Compute distance matrix (cost)
    def distance_matrix_mdf_start_end(bundle_1, bundle_2):
        bundle_1 = set_number_of_points(bundle_1, 2)
        bundle_2 = set_number_of_points(bundle_2, 2)
        return distance_matrix_mdf(bundle_1, bundle_2)

    if distance == 'mdf':
        distance = distance_matrix_mdf
    elif distance == 'mdf_se':
        distance = distance_matrix_mdf_start_end
    else:
        raise ValueError(f"You provided a distance input {distance}, but the \
                         possible options are: 'mdf' or 'mdf_se'")

    cost = distance(bundle1, bundle2)

    # Match and average n1 streamlines based on RLAP
    matched_pairs = np.asarray(linear_sum_assignment(cost)).T

    combined = []
    for ind1, ind2 in matched_pairs:
        stream1 = bundle1[ind1]
        stream2 = bundle2[ind2]

        stream_mean = np.mean([stream1, stream2], axis=0)
        combined.append(stream_mean)

    if comb_method == 'rlap':
        return combined

    # Go through n2-n1 unmatched streamlines
    ind2_matched = matched_pairs[:, 1]
    ind2_unmatched = np.setdiff1d(np.arange(len(bundle2)), ind2_matched)

    if comb_method == 'rlap_closest':
        for ind2 in ind2_unmatched:
            ind1 = np.argmin(cost[:, ind2])

            stream1 = bundle1[ind1]
            stream2 = bundle2[ind2]

            stream_mean = np.mean([stream1, stream2], axis=0)
            combined.append(stream_mean)
    elif comb_method == 'rlap_keep':
        for ind2 in ind2_unmatched:
            combined.append(bundle2[ind2])
    else:
        raise ValueError(f"You provided a bundle combination method \
                         {comb_method}, but the possible options are \
                         'random_pick', 'rlap', 'rlap_keep' or 'rlap_closest'")
    return combined


def compute_atlas_bundle(in_dir, subjects=None, group=None, mid_path='',
                         bundle_names=None, model_bundle_dir=None,
                         out_dir=None, merge_out=False, save_temp=False,
                         n_stream_min=10, n_stream_max=5000, n_point=20,
                         distance='mdf', comb_method='rlap',
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
        Directory with model bundles to be used as a reference to move all
        bundles to a common space. If None, bundles are assumed to be in the
        same space and no registration is performed.
    out_dir : str, optional
        Output directory. If None, the current working directory is used.
    merge_out : boolean, optional
        If True the resulting atlases of all bundles are combined into a single
        file. Default is False.
    save_temp : boolean, optional
        If True the intermediate results of each tree level are saved in a temp
        folder in trk and png formats. Default is False.
    n_stream_min : int, optional
        Bundles with less than ``n_stream_min`` streamlines wont be processed.
        Default is 10.
    n_stream_max : int, optional
        Bundles with more than ``n_stream_max`` streamlines are cropped to have
        that number and speed up the computation. Default is 5000.
    n_point : int, optional
        All streamlines are set to have ``n_point`` points. Default is 20.
    distance : str, optional
        Distance metric to be used to combine bundles. Default is 'mdf'. The
        'mdf_se' distance uses only start/end points of streamlines.
    comb_method : str, optional
        Method used to combine each bundle pair. Default is 'rlap'.
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
        raise TypeError("Provided in_dir input {in_dir} is of type \
                        {type(in_dir)} but it must be a string")
    if type(mid_path) != str:
        raise TypeError("Provided mid_path input {mid_path} is of type \
                        {type(mid_path)} but it must be a string")
    if type(n_point) != int:
        raise TypeError("Provided n_point input {n_point} is of type \
                        {type(n_point)} but it must be an integer")
    if isdir(in_dir) is False:
        raise ValueError("Provided input directory {in_dir} does not exist")
    if out_dir is None:
        out_dir = getcwd()
    if isdir(out_dir) is False:
        raise ValueError("Provided output directory {out_dir} does not exist")
    if n_stream_min < 1:
        raise ValueError("Provided n_stream_min input {n_stream_min) is not \
                         >= 1")
    if n_stream_max < 1:
        raise ValueError("Provided n_stream_max input {n_stream_max} is not \
                         >= 1")
    if n_point < 2:
        raise ValueError("Provided n_point input {n_point} is not >= 2")

    logger.info('Input directory:' + in_dir)
    logger.info('Output directory:' + out_dir)

    # Create temporary folder
    temp_dir = join(out_dir, 'temp')
    if isdir(temp_dir):
        logger.warning("There is already a temp folder in out_dir {out_dir}. \
                       Deleting.")
        rmtree(temp_dir)
    mkdir(temp_dir)

    # Get subjects (from in_dir or a BIDS-like participants.tsv file)
    if subjects is None:
        files = listdir(in_dir)
        subjects = [file for file in files if isdir(join(in_dir, file))]
    else:
        df = pd.read_csv(subjects, delimiter='\t', dtype='object')
        if group is None:
            subjects = list(df.participant)
        else:
            subjects = list(df.loc[df.group == group].participant)
    subjects.sort()  # necessary?
    if len(set(subjects)) < len(subjects):
        raise ValueError("There are duplicated subjects names.")

    logger.info(str(len(subjects)) + " subjects to be processed:")
    logger.info(subjects)

    # Get bundle names (from first subject folder or from tsv file)
    if bundle_names is None:
        bundle_dir = join(in_dir, subjects[0], mid_path)
        logger.info("Retrieving bundle names from " + bundle_dir)
        if isdir(bundle_dir) is False:
            raise ValueError("Path to subject bundles {bundle_dir} does not \
                             exist")

        files = listdir(bundle_dir)
        trk_files = [file for file in files if file.endswith('.trk')]
        bundle_names = [splitext(file)[0] for file in trk_files]
    else:
        df = pd.read_csv(bundle_names, delimiter='\t', dtype='object')
        bundle_names = list(df.iloc[:, 0])
    bundle_names.sort()  # necessary?
    if len(set(bundle_names)) < len(bundle_names):
        raise ValueError("Bundle names cannot be duplicated")

    logger.info(f"{len(bundle_names)} bundles to be processed: {bundle_names}")

    # Create a dictionary with all bundle files
    bundle_files = {}
    for sub in subjects:
        for bundle in bundle_names:
            file = join(in_dir, sub, mid_path, bundle + '.trk')
            bundle_files[(sub, bundle)] = file

    # Get model bundle list
    if model_bundle_dir is None:
        model_bundles = None
    else:
        files = listdir(model_bundle_dir)
        trk_files = [file for file in files if file.endswith('.trk')]
        model_bundle_list = [splitext(file)[0] for file in trk_files]

        if not all(x in model_bundle_list for x in bundle_names):
            raise ValueError("Not all the specified bundles have a model")
        model_bundles = {}
        for bundle in bundle_names:
            model_bundles[bundle] = join(model_bundle_dir, bundle + '.trk')

    # Atlas building starts
    logger.info("Atlasing building started")
    atlas = []
    for bundle in bundle_names:
        logger.info("Processing bundle: " + bundle)

        step_dir = join(temp_dir, bundle, 'step_0')
        makedirs(step_dir)

        # Load model bundle if required
        if model_bundles is not None:
            file = model_bundles[bundle]
            bundle_obj = load_tractogram(file, reference='same',
                                         bbox_valid_check=False)
            model_bundle = bundle_obj.streamlines
            header_model \
                = create_tractogram_header(file, *bundle_obj.space_attributes)

        # Preprocess all bundles and save them as trk
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
                logger.warning(f"{file} has {n_stream} streamlines (< \
                               {n_stream_min}). Discarded.")
                continue
            elif n_stream > n_stream_max:
                streamlines = select_random_set_of_streamlines(streamlines,
                                                               n_stream_max)

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
            if save_temp and has_fury:
                show_bundles([streamlines], f'{file}.png')

        logger.info("Bundle preprocessing finished")

        # Compute pairwise registration tree-structure
        tree, alone, n_reg = get_pairwise_tree(n_item=len(file_list))

        # Go through all tree steps
        for i_step, pairs in enumerate(tree):
            new_file_list = list()

            # Create step folder
            step_dir = join(temp_dir, bundle, 'step_' + str(i_step+1))
            mkdir(step_dir)

            # A lonely bundle goes to the next level
            has_lonely = 0
            if alone[i_step] is not None:
                has_lonely = 1

                file_prev = file_list[alone[i_step]]
                file_new = f'{step_dir}/bundle_0_prev_{alone[i_step]}'
                new_file_list.append(file_new)
                copyfile(f'{file_prev}.trk', f'{file_new}.trk')
                if save_temp and has_fury:
                    copyfile(f'{file_prev}.png', f'{file_new}.png')

            # Register and combine each pair of bundles
            for index, pair in enumerate(pairs):
                i = pair[0]
                j = pair[1]

                logger.info(f"step:{i_step+1}/{len(tree)}" +
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

                # Randomly skip steps if speciffied to get a sharper results
                if skip_pairs and np.random.choice([True, False], 1)[0]:
                    combined = combine_bundles(static, aligned, 'random_pick',
                                               distance)
                else:
                    combined = combine_bundles(static, aligned, comb_method,
                                               distance)

                file = f'{step_dir}/bundle_{index+has_lonely}_prev_{i}_{j}'
                new_file_list.append(file)
                new_tractogram = StatefulTractogram(Streamlines(combined),
                                                    reference=header,
                                                    space=Space.RASMM)
                save_trk(new_tractogram, file + ".trk", bbox_valid_check=False)
                if save_temp and has_fury:
                    show_bundles([static, moving, aligned], f'{file}_reg.png',
                                 colors=[(0, 0, 1), (1, 0, 0), (0, 1, 0)])
                    show_bundles([Streamlines(combined)], f'{file}.png')

            file_list = new_file_list
        save_trk(new_tractogram, f'{out_dir}/{bundle}.trk',
                 bbox_valid_check=False)
        atlas.append(Streamlines(combined))

    if not save_temp:
        rmtree(temp_dir)

    if merge_out:
        atlas_merged = np.concatenate(atlas)
        file = f'{out_dir}/whole_brain.trk'
        new_tractogram = StatefulTractogram(atlas_merged, reference=header,
                                            space=Space.RASMM)
        save_trk(new_tractogram, file, bbox_valid_check=False)
        return atlas, atlas_merged
    return atlas
