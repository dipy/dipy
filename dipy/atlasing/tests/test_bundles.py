import os
import tempfile
import zipfile

import numpy as np
from numpy.testing import assert_equal, assert_raises
import pytest

from dipy.align.bundlemin import distance_matrix_mdf
from dipy.atlasing.bundles import (
    combine_bundles,
    compute_atlas_bundle,
    get_pairwise_tree,
    select_streamlines,
)
from dipy.data import get_fnames
from dipy.io.streamline import (
    Space,
    StatefulTractogram,
    create_tractogram_header,
    load_tractogram,
    save_trk,
)
from dipy.tracking.streamline import (
    Streamlines,
    relist_streamlines,
    select_random_set_of_streamlines,
    unlist_streamlines,
)
from dipy.utils.optpkg import optional_package

pd, has_pandas, _ = optional_package("pandas")
_, has_fury, _ = optional_package("fury")


def test_pairwise_tree():
    # Test n_bundle error
    assert_raises(ValueError, get_pairwise_tree, n_bundle=1)
    assert_raises(TypeError, get_pairwise_tree, n_bundle=2.7182)

    # Define an array of values to test
    n_bundle_list = [2, 4, 5, 15, 21]

    correct_n_step = [1, 2, 3, 4, 5]
    correct_n_pair = [[1], [2, 1], [2, 1, 1], [7, 4, 2, 1], [10, 5, 3, 1, 1]]

    for i, n_bundle in enumerate(n_bundle_list):
        matching, unpaired, n_pair = get_pairwise_tree(n_bundle)
        assert_equal(len(n_pair), correct_n_step[i])
        assert_equal(len(matching), correct_n_step[i])
        assert_equal(len(unpaired), correct_n_step[i])
        assert_equal(n_pair, [match.shape[0] for match in matching])
        assert_equal(n_pair, correct_n_pair[i])
        assert_equal(unpaired.count(0), 0)  # lonely bundle is never the first
        assert_equal(
            [len(np.unique(a)) == a.size for a in matching], [True] * len(matching)
        )


def test_select_streamlines():
    # Create sample streamlines
    ns1 = 20
    ns2 = 10

    bundle1 = Streamlines(np.ones((ns1, 3)))
    bundle2 = Streamlines(np.zeros((ns2, 3)))

    # Test weighted strategy with list input
    result = select_streamlines(bundle1, bundle2, n_out=15, strategy="weighted")
    assert_equal(len(result), 15)
    n_stream_ones = np.any(result, axis=1).sum()
    n_stream_zeros = (~np.any(result, axis=1)).sum()
    assert_equal(n_stream_ones, 10)
    assert_equal(n_stream_zeros, 5)

    # Test random strategy with list input
    result = select_streamlines(bundle1, bundle2, n_out=4, strategy="random")
    assert_equal(len(result), 4)

    # Test error cases
    with assert_raises(TypeError):
        select_streamlines(bundle1, bundle2, n_out=None)

    with assert_raises(ValueError):
        select_streamlines(bundle1, bundle2, n_out=2, strategy="invalid")

    # Test edge cases
    # When n_out is larger than total streamlines
    result = select_streamlines(bundle1, bundle2, n_out=10, strategy="weighted")
    assert_equal(len(result), 10)

    # Test with fixed random seed for reproducibility
    rng = np.random.RandomState(42)
    result1 = select_streamlines(
        bundle1, bundle2, n_out=3, strategy="weighted", rng=rng
    )
    rng = np.random.RandomState(42)
    result2 = select_streamlines(
        bundle1, bundle2, n_out=3, strategy="weighted", rng=rng
    )
    for s1, s2 in zip(result1, result2):
        assert_equal(s1, s2)


def test_combine_bundles():
    example_tracts = get_fnames(name="minimal_bundles")

    with tempfile.TemporaryDirectory() as in_dir:
        # Extract data
        with zipfile.ZipFile(example_tracts, "r") as zip_ref:
            zip_ref.extractall(in_dir)

        # Prepare a pair of bundles with 30 and 40 streamlines
        file1 = os.path.join(in_dir, "sub_1", "AF_L.trk")
        file2 = os.path.join(in_dir, "sub_2", "AF_L.trk")

        bundle_obj1 = load_tractogram(file1, reference="same", bbox_valid_check=False)
        bundle_obj2 = load_tractogram(file2, reference="same", bbox_valid_check=False)

        points, offsets = unlist_streamlines(bundle_obj1.streamlines)
        bundle1 = relist_streamlines(points, offsets)

        points, offsets = unlist_streamlines(bundle_obj2.streamlines)
        bundle2 = relist_streamlines(points, offsets)

        bundle1 = Streamlines(bundle1)
        bundle2 = Streamlines(bundle2)

        n_stream1 = 30
        n_stream2 = 40
        bundle1 = select_random_set_of_streamlines(bundle1, n_stream1)
        bundle2 = select_random_set_of_streamlines(bundle2, n_stream2)

        # Test incorrect parameters
        assert_raises(
            ValueError, combine_bundles, bundle1, bundle2, distance="imaginary_distance"
        )
        assert_raises(
            ValueError,
            combine_bundles,
            bundle1,
            bundle2,
            comb_method="imaginary_method",
        )

        # Test empty bundle
        combined = combine_bundles(bundle1, [])
        assert_equal(len(combined), n_stream1)

        combined = combine_bundles([], bundle2)
        assert_equal(len(combined), n_stream2)

        combined = combine_bundles([], [])
        assert_equal(len(combined), 0)

        # Test random pick (expected streamline number)
        combined = combine_bundles(bundle1, bundle2, "random_pick", n_stream=33)
        assert_equal(len(combined), 33)

        combined = combine_bundles(bundle2, bundle1, "random_pick", n_stream=33)
        assert_equal(len(combined), 33)

        # Test RLAP (expected streamline number and correct averaging)
        for dist in ["mdf", "mdf_se"]:
            # Test RLAP
            combined = combine_bundles(bundle1, bundle2, "rlap", dist)
            assert_equal(len(combined), (n_stream1 + n_stream2) / 2)

            combined = combine_bundles(bundle2, bundle1, "rlap", dist)
            assert_equal(len(combined), (n_stream1 + n_stream2) / 2)

            combined = combine_bundles(bundle1, bundle2, "rlap", dist, n_stream=2000)
            assert_equal(len(combined), n_stream2)

            combined = combine_bundles(bundle1, bundle1, "rlap", dist)
            d = np.diagonal(distance_matrix_mdf(bundle1, combined))
            assert_equal(d, np.zeros(n_stream1))

            combined = combine_bundles(
                bundle1, bundle2, "rlap", dist, n_stream=5, d_max=0.0
            )
            assert_equal(len(combined), 5)

            combined = combine_bundles(bundle1, bundle2, "rlap", dist, d_max=0.0)
            assert_equal(len(combined), (n_stream1 + n_stream2) / 2)

            combined = combine_bundles(
                bundle2, bundle1, "rlap", dist, n_stream=38, d_max=10.0
            )
            assert_equal(len(combined), 38)


# @pytest.mark.skipif(not has_pandas or not has_fury, reason="Requires Pandas and Fury")
@pytest.mark.skip(reason="Skipping test_compute_atlas_bundle")
def test_compute_atlas_bundle():
    example_tracts = get_fnames(name="minimal_bundles")

    with tempfile.TemporaryDirectory() as in_dir:
        with zipfile.ZipFile(example_tracts, "r") as zip_ref:
            zip_ref.extractall(in_dir)

        # Test argument errors
        assert_raises(TypeError, compute_atlas_bundle, in_dir=2)
        assert_raises(TypeError, compute_atlas_bundle, in_dir=in_dir, mid_path=3.14)
        assert_raises(TypeError, compute_atlas_bundle, in_dir=in_dir, n_point=2.78)
        assert_raises(ValueError, compute_atlas_bundle, in_dir="fake_folder")
        assert_raises(
            ValueError, compute_atlas_bundle, in_dir=in_dir, out_dir="fake_folder"
        )
        assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir, n_stream_max=0)
        assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir, n_stream_min=0)
        assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir, n_point=0)

        # Test wrong mid_path
        assert_raises(
            ValueError, compute_atlas_bundle, in_dir=in_dir, mid_path="fake_path"
        )

        # Test bundle duplications
        df = pd.DataFrame(["AF_L", "AF_L"], columns=["bundle"])
        fname = os.path.join(in_dir, "bundles.tsv")
        df.to_csv(fname, sep="\t", index=False)
        assert_raises(
            ValueError, compute_atlas_bundle, in_dir=in_dir, bundle_names=fname
        )

        # Test bundle names and model bundle list mismatch
        df = pd.DataFrame(["AF_L", "AF_R"], columns=["bundle"])
        fname = os.path.join(in_dir, "bundles.tsv")
        df.to_csv(fname, sep="\t", index=False)
        model_bundle_dir = os.path.join(in_dir, "sub_1")  # Use first subject as model
        assert_raises(
            ValueError,
            compute_atlas_bundle,
            in_dir=in_dir,
            bundle_names=fname,
            model_bundle_dir=model_bundle_dir,
        )

        # Test subject duplications
        df = pd.DataFrame({"participant": ["sub_1", "sub_2", "sub_2", "sub_4"]})
        fname = os.path.join(in_dir, "participants.tsv")
        df.to_csv(fname, sep="\t", index=False)
        assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir, subjects=fname)

        # Test default functionality
        with tempfile.TemporaryDirectory() as out_dir:
            atlas, _ = compute_atlas_bundle(in_dir=in_dir, out_dir=out_dir)

            assert_equal(len(atlas), 3)
            assert_equal(os.path.isfile(f"{out_dir}/AF_L.trk"), True)
            assert_equal(os.path.isfile(f"{out_dir}/CST_R.trk"), True)
            assert_equal(os.path.isfile(f"{out_dir}/CC_ForcepsMajor.trk"), True)
            assert_equal(os.path.isdir(f"{out_dir}/temp"), False)

        # Test specific functionalities

        # Subjects and bundles provided as tsv with different groups
        subjects = ["sub_1", "sub_2", "sub_3", "sub_4", "sub_5"]
        group = ["control", "patient", "control", "control", "control"]
        df = pd.DataFrame({"participant": subjects, "group": group})
        subjects = os.path.join(in_dir, "participants.tsv")
        df.to_csv(subjects, sep="\t", index=False)

        df = pd.DataFrame(["AF_L", "CC_ForcepsMajor"], columns=["bundle"])
        bundle_names = os.path.join(in_dir, "bundles.tsv")
        df.to_csv(bundle_names, sep="\t", index=False)

        # One bundle with only 4 streamlines (to be discarded)
        file = f"{in_dir}/sub_1/AF_L.trk"
        bundle_obj = load_tractogram(file, reference="same", bbox_valid_check=False)
        header = create_tractogram_header(file, *bundle_obj.space_attributes)
        bundle = select_random_set_of_streamlines(bundle_obj.streamlines, 4)
        new_tractogram = StatefulTractogram(bundle, reference=header, space=Space.RASMM)
        save_trk(new_tractogram, file, bbox_valid_check=False)

        with tempfile.TemporaryDirectory() as out_dir:
            # temp folder to be removed by the function
            os.mkdir(f"{out_dir}/temp")

            atlas, atlas_merged = compute_atlas_bundle(
                in_dir=in_dir,
                subjects=subjects,
                group="control",
                out_dir=out_dir,
                bundle_names=bundle_names,
                model_bundle_dir=model_bundle_dir,
                save_temp=True,
                merge_out=True,
                skip_pairs=True,
                n_stream_min=5,
                n_stream_max=10,
            )

            assert_equal(len(atlas), 2)
            assert_equal(len(atlas_merged), len(np.concatenate(atlas)))
            assert_equal(os.path.isfile(f"{out_dir}/AF_L.trk"), True)
            assert_equal(os.path.isfile(f"{out_dir}/CC_ForcepsMajor.trk"), True)
            assert_equal(os.path.isfile(f"{out_dir}/whole_brain.trk"), True)

            assert_equal(os.path.isdir(f"{out_dir}/temp"), True)

            temp_files = os.listdir(f"{out_dir}/temp/AF_L/step_0")
            trk_files = [file for file in temp_files if file.endswith(".trk")]
            png_files = [file for file in temp_files if file.endswith(".png")]

            assert_equal(len(trk_files), 3)  # 4 controls - 1 discarded
            assert_equal(len(png_files), 3)


if __name__ == "__main__":
    test_combine_bundles()
