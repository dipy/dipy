import os
import tempfile
import zipfile

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal, assert_raises
import pytest

from dipy.align.bundlemin import distance_matrix_mdf
from dipy.align.streamwarp import bundlewarp
from dipy.atlasing import bundles as atlasing_bundles
from dipy.atlasing.bundles import (
    _load_bundle,
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
    save_trx,
)
from dipy.tracking.streamline import (
    Streamlines,
    relist_streamlines,
    select_random_set_of_streamlines,
    transform_streamlines,
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
        combined = combine_bundles(
            bundle1, bundle2, comb_method="random_pick", n_stream=33
        )
        assert_equal(len(combined), 33)

        combined = combine_bundles(
            bundle2, bundle1, comb_method="random_pick", n_stream=33
        )
        assert_equal(len(combined), 33)

        # Test RLAP (expected streamline number and correct averaging)
        for dist in ["mdf", "mdf_se"]:
            # Test RLAP
            combined = combine_bundles(
                bundle1, bundle2, comb_method="rlap", distance=dist
            )
            assert_equal(len(combined), (n_stream1 + n_stream2) / 2)

            combined = combine_bundles(
                bundle2, bundle1, comb_method="rlap", distance=dist
            )
            assert_equal(len(combined), (n_stream1 + n_stream2) / 2)

            combined = combine_bundles(
                bundle1, bundle2, comb_method="rlap", distance=dist, n_stream=2000
            )
            assert_equal(len(combined), n_stream2)

            combined = combine_bundles(
                bundle1, bundle1, comb_method="rlap", distance=dist
            )
            d = np.diagonal(distance_matrix_mdf(bundle1, combined))
            assert_equal(d, np.zeros(n_stream1))

            combined = combine_bundles(
                bundle1,
                bundle2,
                comb_method="rlap",
                distance=dist,
                n_stream=5,
                d_max=0.0,
            )
            assert_equal(len(combined), 5)

            combined = combine_bundles(
                bundle1, bundle2, comb_method="rlap", distance=dist, d_max=0.0
            )
            assert_equal(len(combined), (n_stream1 + n_stream2) / 2)

            combined = combine_bundles(
                bundle2,
                bundle1,
                comb_method="rlap",
                distance=dist,
                n_stream=38,
                d_max=10.0,
            )
            assert_equal(len(combined), 38)


@pytest.mark.skipif(not has_pandas or not has_fury, reason="Requires Pandas and Fury")
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

        with tempfile.TemporaryDirectory() as out_dir:
            # Test wrong mid_path
            assert_raises(
                ValueError,
                compute_atlas_bundle,
                in_dir=in_dir,
                mid_path="fake_path",
                out_dir=out_dir,
            )

            # Test bundle duplications
            df = pd.DataFrame(["AF_L", "AF_L"], columns=["bundle"])
            fname = os.path.join(in_dir, "bundles.tsv")
            df.to_csv(fname, sep="\t", index=False)
            assert_raises(
                ValueError,
                compute_atlas_bundle,
                in_dir=in_dir,
                out_dir=out_dir,
                bundle_names=fname,
            )

            # Test subject duplications
            df = pd.DataFrame({"participant": ["sub_1", "sub_2", "sub_2", "sub_4"]})
            fname = os.path.join(in_dir, "participants.tsv")
            df.to_csv(fname, sep="\t", index=False)
            assert_raises(
                ValueError,
                compute_atlas_bundle,
                in_dir=in_dir,
                out_dir=out_dir,
                subjects=fname,
            )

        # Test default functionality
        with tempfile.TemporaryDirectory() as out_dir:
            atlas, _ = compute_atlas_bundle(in_dir=in_dir, out_dir=out_dir)

            outputs = os.listdir(out_dir)
            out_dir_name = [x for x in outputs if x.startswith("bundle_atlasing")][0]
            out_dir_trk = os.path.join(out_dir, out_dir_name)

            assert_equal(len(atlas), 3)
            assert_equal(os.path.isfile(f"{out_dir_trk}/AF_L.trk"), True)
            assert_equal(os.path.isfile(f"{out_dir_trk}/CST_R.trk"), True)
            assert_equal(os.path.isfile(f"{out_dir_trk}/CC_ForcepsMajor.trk"), True)
            assert_equal(os.path.isdir(f"{out_dir_trk}/temp"), False)

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
            atlas, atlas_merged = compute_atlas_bundle(
                in_dir=in_dir,
                subjects=subjects,
                group="control",
                out_dir=out_dir,
                bundle_names=bundle_names,
                save_temp=True,
                merge_out=True,
                skip_pairs=True,
                n_stream_min=5,
                n_stream_max=10,
            )

            outputs = os.listdir(out_dir)
            out_dir_name = [x for x in outputs if x.startswith("bundle_atlasing")][0]
            out_dir_trk = os.path.join(out_dir, out_dir_name)
            temp_dir = os.path.join(out_dir_trk, "temp")

            assert_equal(len(atlas), 2)
            assert_equal(len(atlas_merged), len(np.concatenate(atlas)))
            assert_equal(os.path.isfile(f"{out_dir_trk}/AF_L.trk"), True)
            assert_equal(os.path.isfile(f"{out_dir_trk}/CC_ForcepsMajor.trk"), True)
            assert_equal(os.path.isfile(f"{out_dir_trk}/whole_brain.trk"), True)

            temp_files = os.listdir(f"{temp_dir}/AF_L/step_0")
            trk_files = [file for file in temp_files if file.endswith(".trk")]
            png_files = [file for file in temp_files if file.endswith(".png")]

            assert_equal(len(trk_files), 3)  # 4 controls - 1 discarded
            assert_equal(len(png_files), 9)  # three views * three bundles

        # Check BundleWarp and SLR also work
        with tempfile.TemporaryDirectory() as out_dir:
            atlas, atlas_merged = compute_atlas_bundle(
                in_dir=in_dir,
                subjects=subjects,
                group="control",
                out_dir=out_dir,
                bundle_names=bundle_names,
                save_temp=False,
                n_stream_min=5,
                n_stream_max=10,
                reg_method="bundlewarp",
            )


def test_load_bundle():
    example_tracts = get_fnames(name="minimal_bundles")

    with tempfile.TemporaryDirectory() as in_dir:
        with zipfile.ZipFile(example_tracts, "r") as zip_ref:
            zip_ref.extractall(in_dir)

        trk_file = os.path.join(in_dir, "sub_1", "AF_L.trk")
        bundle, header = _load_bundle(trk_file, return_header=True)

        # always a Streamlines, always float64, whatever the file held
        assert isinstance(bundle, Streamlines)
        assert_equal(bundle[0].dtype, np.float64)
        assert len(bundle) > 0

        # the same bundle stored as TRX must load to the same streamlines
        sft = load_tractogram(trk_file, reference="same", bbox_valid_check=False)
        trx_file = os.path.join(in_dir, "AF_L.trx")
        save_trx(sft, trx_file, bbox_valid_check=False)

        bundle_trx, header_trx = _load_bundle(trx_file, return_header=True)
        assert isinstance(bundle_trx, Streamlines)
        assert_equal(bundle_trx[0].dtype, np.float64)
        assert_equal(len(bundle_trx), len(bundle))
        assert_array_almost_equal(bundle_trx[0], bundle[0], decimal=3)

        # a TRX input still yields a TRK header, since atlases are saved as TRK
        assert_equal(sorted(dict(header_trx)), sorted(dict(header)))


def test_load_bundle_empty():
    example_tracts = get_fnames(name="minimal_bundles")

    with tempfile.TemporaryDirectory() as in_dir:
        with zipfile.ZipFile(example_tracts, "r") as zip_ref:
            zip_ref.extractall(in_dir)

        trk_file = os.path.join(in_dir, "sub_1", "AF_L.trk")
        header = create_tractogram_header(
            trk_file,
            *load_tractogram(
                trk_file, reference="same", bbox_valid_check=False
            ).space_attributes,
        )
        empty_file = os.path.join(in_dir, "empty.trk")
        save_trk(
            StatefulTractogram(Streamlines([]), reference=header, space=Space.RASMM),
            empty_file,
            bbox_valid_check=False,
        )

        bundle = _load_bundle(empty_file)
        assert isinstance(bundle, Streamlines)
        assert_equal(len(bundle), 0)


@pytest.mark.skipif(not has_pandas, reason="Requires Pandas")
def test_bundlewarp_return_order():
    # compute_atlas_bundle relies on bundlewarp returning the warped MOVING
    # bundle first. Different sizes make the two arguments distinguishable, so
    # this fails if the order is ever swapped.
    example_tracts = get_fnames(name="minimal_bundles")

    with tempfile.TemporaryDirectory() as in_dir:
        with zipfile.ZipFile(example_tracts, "r") as zip_ref:
            zip_ref.extractall(in_dir)

        static = _load_bundle(os.path.join(in_dir, "sub_1", "AF_L.trk"))
        moving = _load_bundle(os.path.join(in_dir, "sub_2", "AF_L.trk"))
        static = Streamlines(static[:12])
        moving = Streamlines(moving[:7])

        deformed, moving_aligned, _, _, _ = bundlewarp(static, moving, alpha=0.5)

        assert_equal(len(deformed), len(moving))
        assert_equal(len(moving_aligned), len(moving))


def _mock_mni_reference(monkeypatch, in_dir, ref_dir, bundle_names):
    # Use a shifted copy of a subject as reference to avoid downloading the
    # HCP-842 atlas. Its dimensions differ so the output space can be identified.
    os.makedirs(os.path.join(ref_dir, "bundles"))
    shift = np.eye(4)
    shift[:3, 3] = [40, -30, 25]
    whole_brain = Streamlines()

    def save(streamlines, file):
        sft = StatefulTractogram(streamlines, reference=header, space=Space.RASMM)
        save_trk(sft, file, bbox_valid_check=False)

    for name in bundle_names:
        file = os.path.join(in_dir, "sub_1", f"{name}.trk")
        sft = load_tractogram(file, reference="same", bbox_valid_check=False)
        affine, dimensions, voxel_sizes, voxel_order = sft.space_attributes
        dimensions = np.asarray(dimensions) + 7
        header = create_tractogram_header(
            file, affine, dimensions, voxel_sizes, voxel_order
        )
        bundle = Streamlines(transform_streamlines(sft.streamlines, shift))
        whole_brain.extend(bundle)
        save(bundle, os.path.join(ref_dir, "bundles", f"{name}.trk"))

    whole_brain_file = os.path.join(ref_dir, "whole_brain_MNI.trk")
    save(whole_brain, whole_brain_file)
    pattern = os.path.join(ref_dir, "bundles", "*.trk")

    monkeypatch.setattr(atlasing_bundles, "fetch_30_bundle_atlas_hcp842", lambda: None)
    monkeypatch.setattr(
        atlasing_bundles,
        "get_bundle_atlas_hcp842",
        lambda size: (whole_brain_file, pattern),
    )
    return dimensions


def _centre(streamlines):
    return np.mean(np.concatenate(list(streamlines)), axis=0)


def _output_dir(out_dir):
    name = [x for x in os.listdir(out_dir) if x.startswith("bundle_atlasing")][0]
    return os.path.join(out_dir, name)


def _dimensions(file):
    sft = load_tractogram(file, reference="same", bbox_valid_check=False)
    return sft.dimensions


def test_compute_atlas_bundle_mni(monkeypatch):
    example_tracts = get_fnames(name="minimal_bundles")
    names = ["AF_L", "CST_R", "CC_ForcepsMajor"]
    kwargs = {"bundle_names": names, "n_stream_max": 50}

    with (
        tempfile.TemporaryDirectory() as in_dir,
        tempfile.TemporaryDirectory() as ref_dir,
    ):
        with zipfile.ZipFile(example_tracts, "r") as zip_ref:
            zip_ref.extractall(in_dir)

        # The reference does not have CC_ForcepsMajor
        ref_dims = _mock_mni_reference(monkeypatch, in_dir, ref_dir, names[:2])
        ref = {n: _load_bundle(f"{ref_dir}/bundles/{n}.trk") for n in names[:2]}

        # Test argument errors
        for kwarg in [
            {"final_transform": "similarity"},
            {"final_registration": "imaginary_method"},
        ]:
            assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir, **kwarg)

        # No bundle has a reference
        with tempfile.TemporaryDirectory() as out_dir:
            assert_raises(
                ValueError,
                compute_atlas_bundle,
                in_dir=in_dir,
                out_dir=out_dir,
                bundle_names=names[2:],
                final_registration="bundle",
            )

        # Default: the atlas stays where it was built
        with tempfile.TemporaryDirectory() as out_dir:
            atlas, _ = compute_atlas_bundle(in_dir=in_dir, out_dir=out_dir, **kwargs)
            out = _output_dir(out_dir)
            assert not os.path.isdir(f"{out}/build_space")
            native_dims = _dimensions(f"{out}/AF_L.trk")
            assert not np.array_equal(native_dims, ref_dims)
            assert np.linalg.norm(_centre(atlas[0]) - _centre(ref["AF_L"])) > 30

        # Per bundle registration
        with tempfile.TemporaryDirectory() as out_dir:
            atlas, merged = compute_atlas_bundle(
                in_dir=in_dir,
                out_dir=out_dir,
                merge_out=True,
                final_registration="bundle",
                **kwargs,
            )
            out = _output_dir(out_dir)
            for i, name in enumerate(names[:2]):
                assert np.linalg.norm(_centre(atlas[i]) - _centre(ref[name])) < 10
                assert_equal(_dimensions(f"{out}/{name}.trk"), ref_dims)
                assert_equal(_dimensions(f"{out}/build_space/{name}.trk"), native_dims)

                # the default rigid transformation does not change the size
                affine = np.loadtxt(f"{out}/build_space/{name}_to_mni_affine.txt")
                singular_values = np.linalg.svd(affine[:3, :3], compute_uv=False)
                assert_array_almost_equal(singular_values, np.ones(3), decimal=5)

            # A bundle without reference stays as built and is not merged
            assert_equal(_dimensions(f"{out}/CC_ForcepsMajor.trk"), native_dims)
            assert not os.path.exists(f"{out}/build_space/CC_ForcepsMajor.trk")
            assert_equal(len(merged), len(atlas[0]) + len(atlas[1]))

        # Whole brain registration enables merge_out and includes all the bundles,
        # also the ones without reference
        with tempfile.TemporaryDirectory() as out_dir:
            atlas, merged = compute_atlas_bundle(
                in_dir=in_dir,
                out_dir=out_dir,
                final_registration="whole_brain",
                final_transform="affine",
                **kwargs,
            )
            out = _output_dir(out_dir)
            assert merged is not None
            assert_equal(len(merged), sum(len(bundle) for bundle in atlas))
            assert_equal(_dimensions(f"{out}/whole_brain.trk"), ref_dims)
            assert_equal(_dimensions(f"{out}/AF_L.trk"), ref_dims)
            assert_equal(_dimensions(f"{out}/CC_ForcepsMajor.trk"), ref_dims)
            assert_equal(_dimensions(f"{out}/build_space/whole_brain.trk"), native_dims)
            assert os.path.isfile(f"{out}/build_space/whole_brain_to_mni_affine.txt")
            assert np.linalg.norm(_centre(atlas[0]) - _centre(ref["AF_L"])) < 10
