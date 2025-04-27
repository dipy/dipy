"""Atlasing module: utilities to compute population-specific streamline-based bundle
atlases.

Available functions:
    get_pairwise_tree: computes the matching pairs for a given number of bundles.

    combine_bundle: combines two bundles into a single one using different
    streamline combination methods.

    compute_bundle_atlas: given a list of input bundles computes individual
    atlases for each white matter tract.
"""

from datetime import datetime
import logging
from os import getcwd, listdir, makedirs, mkdir
from os.path import isdir, join, splitext
from shutil import copyfile, rmtree

import numpy as np
from scipy.optimize import linear_sum_assignment

from dipy.align.bundlemin import distance_matrix_mdf
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.align.streamwarp import bundlewarp
from dipy.io.streamline import (
    Space,
    StatefulTractogram,
    create_tractogram_header,
    load_trk,
    save_trk,
)
from dipy.segment.clustering import QuickBundles, qbx_and_merge
from dipy.tracking.streamline import (
    Streamlines,
    orient_by_streamline,
    select_random_set_of_streamlines,
    set_number_of_points,
)
from dipy.utils.optpkg import optional_package
from dipy.viz.streamline import show_bundles

pd, has_pd, _ = optional_package("pandas")
_, has_fury, _ = optional_package("fury")


logger = logging.getLogger(__name__)


def get_pairwise_tree(n_bundle, seed=None):
    """Pairwise tree structure calculation.

    Constructs a pairwise tree by randomly matching the indexes of a given
    number of bundles. The computed index structure is intended to be used for
    atlasing, where the bundles are combined in pairs until a single bundle
    (atlas) is obtained.

    Parameters
    ----------
    n_bundle : int
        Number of bundles to be matched.
    seed : float, optional
        Seed for reproducibility. Default is None.

    Returns
    -------
    matched_pairs : list of array
        Each element in list is an array of (N x 2) with the indexes of the
        N items to be matched in a certain level.
    unpaired : list of array
        Indexes of the items not combined at a certain level when the number of
        items to be matched is odd. When it is even this value is set to None.
    n_pair : list of int
        Number of pairwise matches at each level of the tree.
    """
    if not isinstance(n_bundle, int):
        raise TypeError(f"n_bundle must be an integer and not {type(n_bundle)}")
    if n_bundle <= 1:
        raise ValueError(f"n_bundle must be greater than 1 but is {n_bundle}")

    np.random.seed(seed)

    matched_pairs = []
    unpaired = []
    n_pair = []

    while n_bundle > 1:
        # Define indexes
        index = np.arange(n_bundle)
        # Compute the number of pairs
        n_pair.append(np.floor(n_bundle / 2).astype("int"))
        # Shuffle the indexes
        index = np.random.permutation(n_bundle)
        # If n_bundle is odd do not pair one of the bundles
        if np.mod(n_bundle, 2) == 1:
            # avoid removing again an item twice (index == 0)
            unpaired_idx = np.random.randint(1, n_bundle)
            index = np.delete(index, np.where(index == unpaired_idx))
            unpaired.append(unpaired_idx)
        else:
            unpaired.append(None)
        # Generate pairwise index matrix
        index = np.reshape(index, (n_pair[-1], 2))
        # Update bundle number
        n_bundle = np.ceil(n_bundle / 2).astype("int")

        matched_pairs.append(index)

    return matched_pairs, unpaired, n_pair


def select_streamlines(bundle1, bundle2, n_out, strategy="weighted", rng=None):
    """Select and combine streamlines from two bundles.

    Parameters
    ----------
    bundle1 : list or Streamlines
        First bundle of streamlines to select from.
    bundle2 : list or Streamlines
        Second bundle of streamlines to select from.
    n_out : int
        Number of streamlines to select in total.
    strategy : {'weighted', 'random'}, optional
        Strategy to use when selecting streamlines:
        - 'weighted': Select streamlines proportionally to bundle sizes
        - 'random': Randomly select from combined bundles
        Default is 'weighted'.
    rng : RandomState or None, optional
        Random number generator to use for streamline selection.
        Default is None.

    Returns
    -------
    list or Streamlines
        Selected streamlines combined from both bundles.

    Notes
    -----
    For the 'weighted' strategy, the number of streamlines selected from each
    bundle is proportional to their original sizes. For example, if bundle1 has
    20 streamlines and bundle2 has 10, and n_out=15, then 10 streamlines will
    be selected from bundle1 and 5 from bundle2.
    """
    n1 = len(bundle1)
    n2 = len(bundle2)

    if n_out is None:
        raise TypeError("n_out must be a numeric value")

    if strategy == "weighted":
        ns1 = np.round(n1 / (n1 + n2) * n_out).astype(int)
        ns2 = n_out - ns1

        bundle1 = select_random_set_of_streamlines(bundle1, ns1, rng=rng)
        bundle2 = select_random_set_of_streamlines(bundle2, ns2, rng=rng)

        if ns1 > 0 and ns2 > 0:
            if isinstance(bundle1, list):
                return np.concatenate((bundle1, bundle2))
            else:
                bundle1.extend(bundle2)
                return bundle1
        elif ns1 > 0 and ns2 == 0:
            return bundle1
        elif ns1 == 0 and ns2 > 0:
            return bundle2
        else:
            raise ValueError("Not enough streamlines sampled")

    elif strategy == "random":
        if isinstance(bundle1, list):
            bundles = np.concatenate((bundle1, bundle2))
        else:
            bundles = bundle1.copy()
            bundles.extend(bundle2)

        return select_random_set_of_streamlines(bundles, n_out, rng=rng)
    else:
        raise ValueError("Invalid streamline selection strategy.")


def combine_bundles(
    bundle1,
    bundle2,
    comb_method="rlap",
    distance="mdf",
    n_stream="mean",
    d_max=1000.0,
):
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
        Default is 'mean'.
    d_max : float, optional
        Maximum distance to allow averaging. Higher numbers result in smoother atlases.
        Default is 1000.

    Returns
    -------
    combined : list
        Streamline coordinates of the combined bundle as a list of 2D ndarrays
        of shape[-1]==3.
    """

    if comb_method not in ["rlap", "random_pick"]:
        raise ValueError("Invalid streamline combination method")

    # Set as bundle 1 the bundle with less streamlines
    if len(bundle2) < len(bundle1):
        aux = bundle1.copy()
        bundle1 = bundle2
        bundle2 = aux

    n_stream1 = len(bundle1)
    n_stream2 = len(bundle2)

    if n_stream1 == 0:
        logger.info("Bundle 1 is empty. Returning bundle 2.")
        return bundle2

    if isinstance(n_stream, int):
        n_stream = np.min([n_stream, n_stream2])
    else:
        if n_stream == "min":
            n_stream = np.min([n_stream1, n_stream2])
        elif n_stream == "mean":
            n_stream = np.round(np.mean([n_stream1, n_stream2])).astype(int)
        elif n_stream == "max":
            n_stream = np.max([n_stream1, n_stream2])
        else:
            raise ValueError("n_stream must be ['min','mean','max'] or an int")

    # If random_pick just merge all streamlines and pick n_stream randomly
    if comb_method == "random_pick":
        return select_streamlines(bundle1, bundle2, n_stream, "weighted")

    # Reorientation based on centroid
    bundle1 = orient_by_streamline(bundle1, bundle1[0])
    qb = QuickBundles(threshold=50.0)
    centroid = qb.cluster(bundle2).centroids[0]

    bundle1 = orient_by_streamline(bundle1, centroid)
    bundle2 = orient_by_streamline(bundle2, centroid)

    # Step 1: streamline matching based on RLAP
    combined = []

    if distance == "mdf":
        cost = distance_matrix_mdf(bundle1, bundle2)
    elif distance == "mdf_se":
        bundle1_se = set_number_of_points(bundle1, 2)
        bundle2_se = set_number_of_points(bundle2, 2)
        cost = distance_matrix_mdf(bundle1_se, bundle2_se)
    else:
        raise ValueError("Incorrect distance metric")

    matched_pairs = np.asarray(linear_sum_assignment(cost)).T

    unmatched = np.setdiff1d(np.arange(n_stream2), matched_pairs[:, 1])

    # Step 2: streamline combination
    outliers1 = []
    outliers2 = []
    for ind1, ind2 in matched_pairs:
        stream1 = bundle1[ind1]
        stream2 = bundle2[ind2]

        # Discard too distant streamline pairs
        if cost[ind1, ind2] > d_max:
            outliers1.append(ind1)
            outliers2.append(ind2)
            continue

        stream_mean = np.mean([stream1, stream2], axis=0)
        combined.append(stream_mean)
    n_combined = len(combined)

    logger.info(f"Discarded {len(outliers1)}/{n_stream1} streamlines as outliers.")

    # Step 3: remaining streamlines
    if n_combined > n_stream:
        combined = select_random_set_of_streamlines(combined, n_stream)
    elif n_combined < n_stream:
        n_extra = n_stream - n_combined

        remain1 = bundle1[outliers1] if len(outliers1) > 0 else []
        remain2 = bundle2[np.hstack((unmatched, outliers2)).astype("int").tolist()]

        extra = select_streamlines(remain1, remain2, n_extra, "weighted")
        extra = list(extra)  # matrix to list

        combined = [*combined, *extra]

        logger.info(f"Added extra {n_extra} streamlines")

    return combined


def _load_bundle(file, return_header=False):
    """Load a tractography bundle from a .trk file.

    Parameters
    ----------
    file : str
        Path to the .trk file containing the bundle.
    return_header : bool, optional
        If True, returns both the bundle and header information.
        Default is False.

    Returns
    -------
    bundle : Streamlines or StatefulTractogram
        The loaded streamlines. If the file contains streamlines, returns them
        as a Streamlines object after unlisting and relisting. If empty, returns
        the original StatefulTractogram object.
    header : TrkFile
        Only returned if return_header is True. Contains the tractogram header
        information including space attributes.

    Notes
    -----
    This is a private function used internally by the atlasing module to load
    and process bundle files consistently.
    """
    bundle_obj = load_trk(file, reference="same", bbox_valid_check=False)

    if len(bundle_obj.streamlines) > 0:
        bundle = list(bundle_obj.streamlines)
    else:
        bundle = bundle_obj

    if return_header:
        header = create_tractogram_header(file, *bundle_obj.space_attributes)
        return bundle, header

    return bundle


def compute_atlas_bundle(
    in_dir,
    subjects=None,
    group=None,
    mid_path="",
    bundle_names=None,
    out_dir=None,
    merge_out=False,
    save_temp=False,
    n_stream_min=10,
    n_stream_max=5000,
    n_point=20,
    reg_method="hslr",
    x0="affine",
    distance="mdf",
    comb_method="rlap",
    skip_pairs=False,
    d_max=1000.0,
    qbx_thr=5,
    alpha=0.5,
):
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
    bundle_names : str or list of str, optional
        If a string, it is interpreted as the path to a tsv file with the names
        of the bundles to be processed. If a list of strings, it is interpreted
        as the names of the bundles to be processed. If None, all trk files of
        the first subject will be considered as bundle_names.
    out_dir : str, optional
        Output directory. If None, the current working directory is used.
    merge_out : boolean, optional
        If True the resulting atlases of all bundles are combined into a single
        file. Default is False.
    save_temp : boolean, optional
        If True the intermediate results of each tree level are saved in a temp
        folder in trk and png formats. Default is False.
    n_stream_min : int, optional
        Bundles with less than ``n_stream_min`` streamlines won't be processed.
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
    reg_method : str, optional
        Registration method to be used ('slr', 'hslr', 'bundlewarp').
        Default is 'hslr'.
    skip_pairs : boolean, optional
        If true bundle combination steps are randomly skipped. This helps to
        obtain a sharper result. Default is False.
    d_max : float, optional
        Maximum distance to allow averaging. Higher numbers result in smoother atlases.
        Default is 1000.
    qbx_thr : float, optional
        Threshold for QuickBundles clustering. Default is 5.
    alpha : float, optional
        Controls deformation strength of bundlewarp. Lower values allow more
        deformation. Only used if reg_method is 'bundlewarp'. Default is 0.5.

    Returns
    -------
    atlas : list of Streamlines
        A list with the computed atlas bundles.
    atlas_merged : Streamlines
        A single bundle containing all the computed atlas bundles together.
    """
    if not isinstance(in_dir, str):
        raise TypeError(f"in_dir {in_dir} is {type(in_dir)} but must be a string")
    if not isinstance(mid_path, str):
        raise TypeError(f"mid_path {mid_path} is {type(mid_path)} but must be a string")
    if not isinstance(n_point, int):
        raise TypeError(f"n_point {n_point} is {type(n_point)} but must be an integer")
    if not isdir(in_dir):
        raise ValueError(f"in_dir {in_dir} does not exist")
    if out_dir is None:
        out_dir = getcwd()
    if not isdir(out_dir):
        raise ValueError(f"out_dir {out_dir} does not exist")
    if n_stream_min < 1:
        raise ValueError(f"n_stream_min {n_stream_min} is not >= 1")
    if n_stream_max < 1:
        raise ValueError(f"n_stream_max {n_stream_max} is not >= 1")
    if n_point < 2:
        raise ValueError(f"n_point {n_point} is not >= 2")
    if reg_method not in ["slr", "hslr", "bundlewarp"]:
        raise ValueError("reg_method value not supported")

    logger.info("Input directory:" + in_dir)
    logger.info("Output directory:" + out_dir)

    # Create temporary folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = join(out_dir, f"bundle_atlasing_{timestamp}")
    temp_dir = join(out_dir, "temp")
    makedirs(out_dir, exist_ok=True)
    makedirs(temp_dir, exist_ok=True)

    # Get subjects (from in_dir or a BIDS-like participants.tsv file)
    if subjects is None:
        files = listdir(in_dir)
        subjects = [file for file in files if isdir(join(in_dir, file))]
    else:
        df = pd.read_csv(subjects, delimiter="\t", dtype="object")
        if group is not None:
            df = df[df.group == group]
        subjects = df.participant.tolist()
    if len(set(subjects)) < len(subjects):
        raise ValueError("There are duplicated subjects names.")

    logger.info(f"Number of subjects to be processed: {len(subjects)}")
    logger.debug(subjects)

    # Get bundle names (from first subject folder or from tsv file)
    if bundle_names is None:
        bundle_dir = join(in_dir, subjects[0], mid_path)
        logger.info("Retrieving bundle names from " + bundle_dir)
        if isdir(bundle_dir) is False:
            raise ValueError(f"Path to subject bundles {bundle_dir} does not exist")

        files = listdir(bundle_dir)
        trk_files = [file for file in files if file.endswith(".trk")]
        bundle_names = [splitext(file)[0] for file in trk_files]
    elif not isinstance(bundle_names, list):
        df = pd.read_csv(bundle_names, delimiter="\t", dtype="object")
        bundle_names = list(df.iloc[:, 0])

    if len(set(bundle_names)) < len(bundle_names):
        raise ValueError("Bundle names cannot be duplicated")

    logger.info(f"{len(bundle_names)} bundles to be processed: {bundle_names}")

    # Create a dictionary with all bundle files
    bundle_files = {}
    for sub in subjects:
        for bundle in bundle_names:
            file = join(in_dir, sub, mid_path, bundle + ".trk")
            bundle_files[(sub, bundle)] = file

    # Atlas building starts
    logger.info("Atlas building started")
    atlas = []
    for bundle in bundle_names:
        logger.info("Processing bundle: " + bundle)

        step_dir = join(temp_dir, bundle, "step_0")
        makedirs(step_dir)

        # Preprocess all bundles and save them as trk
        file_list = []
        for i, sub in enumerate(subjects):
            file = bundle_files[(sub, bundle)]
            streamlines, header = _load_bundle(file, return_header=True)

            n_stream = len(streamlines)
            if n_stream < n_stream_min:
                logger.warning(
                    f"Discarding {file}: has {n_stream} streamlines (< {n_stream_min})"
                )
                continue

            streamlines = select_random_set_of_streamlines(streamlines, n_stream_max)
            streamlines = set_number_of_points(streamlines, n_point)

            file = f"{step_dir}/bundle_{i}_prev_{sub}"
            file_list.append(file)
            new_tractogram = StatefulTractogram(
                streamlines, reference=header, space=Space.RASMM
            )
            save_trk(new_tractogram, f"{file}.trk", bbox_valid_check=False)
            if save_temp and has_fury:
                for view in ["sagittal", "axial", "coronal"]:
                    show_bundles(
                        [streamlines],
                        interactive=False,
                        save_as=f"{file}_{view}.png",
                        view=view,
                    )

        logger.info("Bundle preprocessing finished")

        # Compute pairwise registration tree-structure
        tree, unpaired, n_reg = get_pairwise_tree(n_bundle=len(file_list))

        # Go through all tree steps
        for i_step, pairs in enumerate(tree):
            new_file_list = []

            # Create step folder
            step_dir = join(temp_dir, bundle, "step_" + str(i_step + 1))
            mkdir(step_dir)

            # An unpaired bundle goes to the next level
            has_unpaired = 0
            if unpaired[i_step] is not None:
                has_unpaired = 1

                file_prev = file_list[unpaired[i_step]]
                file_new = f"{step_dir}/bundle_0_prev_{unpaired[i_step]}"
                new_file_list.append(file_new)
                copyfile(f"{file_prev}.trk", f"{file_new}.trk")

                if save_temp and has_fury:
                    for view in ["sagittal", "axial", "coronal"]:
                        copyfile(f"{file_prev}_{view}.png", f"{file_new}_{view}.png")

            # Register and combine each pair of bundles
            for index, pair in enumerate(pairs):
                i = pair[0]
                j = pair[1]

                logger.info(
                    f"step:{i_step + 1}/{len(tree)}"
                    + f" pair:{index + 1}/{n_reg[i_step]}"
                )

                static, header = _load_bundle(file_list[i] + ".trk", return_header=True)
                moving = _load_bundle(file_list[j] + ".trk")

                if reg_method in ["slr", "hslr"]:
                    # BundleWarp does not work with clusters
                    moving_clusters = qbx_and_merge(moving, thresholds=[qbx_thr])
                    static_clusters = qbx_and_merge(static, thresholds=[qbx_thr])
                    moving_centroids = moving_clusters.centroids
                    static_centroids = static_clusters.centroids

                    if reg_method == "slr":
                        slr = StreamlineLinearRegistration(x0=x0)
                        srm = slr.optimize(
                            static=static_centroids, moving=moving_centroids
                        )
                        aligned = srm.transform(moving)

                    elif reg_method == "hslr":
                        hslr = StreamlineLinearRegistration(x0=x0, halfway=True)
                        srm = hslr.optimize(moving_centroids, static_centroids)
                        [aligned, static] = srm.transform(moving, static)

                    static = Streamlines(static)
                    aligned = Streamlines(aligned)

                elif reg_method == "bundlewarp":
                    static, aligned, _, _, _ = bundlewarp(
                        Streamlines(static), Streamlines(moving), alpha=alpha
                    )

                # Randomly skip steps if specified to get sharper results
                if skip_pairs and np.random.choice([True, False], 1)[0]:
                    combined = combine_bundles(
                        static,
                        aligned,
                        "random_pick",
                        distance,
                        n_stream=n_stream_max,
                        d_max=d_max,
                    )
                else:
                    combined = combine_bundles(
                        static,
                        aligned,
                        comb_method,
                        distance,
                        n_stream=n_stream_max,
                        d_max=d_max,
                    )

                file = f"{step_dir}/bundle_{index + has_unpaired}_prev_{i}_{j}"
                new_file_list.append(file)
                new_tractogram = StatefulTractogram(
                    Streamlines(combined), reference=header, space=Space.RASMM
                )
                save_trk(new_tractogram, file + ".trk", bbox_valid_check=False)
                if save_temp and has_fury:
                    for view in ["sagittal", "axial", "coronal"]:
                        show_bundles(
                            [static, aligned, combined],
                            interactive=False,
                            colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                            save_as=f"{file}_{view}_parents.png",
                            view=view,
                        )
                        show_bundles(
                            [Streamlines(combined)],
                            interactive=False,
                            colors=[(0, 0, 1)],
                            save_as=f"{file}_{view}.png",
                            view=view,
                        )

            file_list = new_file_list
        save_trk(new_tractogram, f"{out_dir}/{bundle}.trk", bbox_valid_check=False)
        atlas.append(Streamlines(combined))

    if not save_temp:
        rmtree(temp_dir)

    atlas_merged = None
    if merge_out:
        atlas_merged = np.concatenate(atlas)
        file = f"{out_dir}/whole_brain.trk"
        new_tractogram = StatefulTractogram(
            atlas_merged, reference=header, space=Space.RASMM
        )
        save_trk(new_tractogram, file, bbox_valid_check=False)

    return atlas, atlas_merged
