import os
import zipfile
import numpy as np
from numpy.testing import assert_equal, assert_raises, run_module_suite
import pandas as pd
from tempfile import TemporaryDirectory

from dipy.atlasing.bundles import (combine_bundles, compute_atlas_bundle,
                                   get_pairwise_tree)
from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import (relist_streamlines,
                                      select_random_set_of_streamlines,
                                      unlist_streamlines)


def setup_module():
    global in_dir
    # Load toy data
    in_dir = TemporaryDirectory()
    example_tracts = get_fnames('minimal_bundles')
    with zipfile.ZipFile(example_tracts, 'r') as zip_ref:
        zip_ref.extractall(in_dir.name)


def test_pairwise_tree():
    # Test n_bundles error
    assert_raises(ValueError, get_pairwise_tree, n_item=1)
    assert_raises(TypeError, get_pairwise_tree, n_item=2.7182)

    # Define an array of values to test
    n_item_list = [2, 4, 5, 15, 21]

    expected_n_step = [1, 2, 3, 4, 5]
    expected_n_reg = [[1], [2, 1], [2, 1, 1], [7, 4, 2, 1], [10, 5, 3, 1, 1]]

    for i, n_item in enumerate(n_item_list):
        matching, alone, n_reg = get_pairwise_tree(n_item)
        assert_equal(len(n_reg), expected_n_step[i])
        assert_equal(len(matching), expected_n_step[i])
        assert_equal(len(alone), expected_n_step[i])
        assert_equal(n_reg, [match.shape[0] for match in matching])
        assert_equal(n_reg, expected_n_reg[i])
        assert_equal(alone.count(0), 0)  # lonely bundle is never the first
        assert_equal([len(np.unique(a)) == a.size for a in matching],
                     [True]*len(matching))


def test_combine_bundle():
    # Prepare a pair of bundles with 30 and 50 streamlines
    bundle_obj1 = load_tractogram(f'{in_dir.name}/sub_1/AF_L.trk',
                                  reference='same', bbox_valid_check=False)
    points, offsets = unlist_streamlines(bundle_obj1.streamlines)
    bundle1 = relist_streamlines(points, offsets)
    n_stream1 = 30
    bundle1 = select_random_set_of_streamlines(bundle1, n_stream1)

    bundle_obj2 = load_tractogram(f'{in_dir.name}/sub_2/AF_L.trk',
                                  reference='same', bbox_valid_check=False)
    points, offsets = unlist_streamlines(bundle_obj2.streamlines)
    bundle2 = relist_streamlines(points, offsets)

    # Test incorrect parameters
    assert_raises(ValueError, combine_bundles, bundle1, bundle2,
                  distance='imaginary_distance')
    assert_raises(ValueError, combine_bundles, bundle1, bundle2,
                  comb_method='imaginary_method')

    # Test expected streamline number and correct averaging
    for dist in ['mdf', 'mdf_se']:
        # Test RLAP
        combined = combine_bundles(bundle1, bundle2, 'rlap', dist)
        assert_equal(len(combined), 30)
        combined = combine_bundles(bundle2, bundle1, 'rlap', dist)
        assert_equal(len(combined), 30)
        combined = combine_bundles(bundle1, bundle1, 'rlap', dist)
        assert_equal(combined, bundle1)
        # Test RLAP + keep
        combined = combine_bundles(bundle1, bundle2, 'rlap_keep', dist)
        assert_equal(len(combined), 50)
        combined = combine_bundles(bundle2, bundle1, 'rlap_keep', dist)
        assert_equal(len(combined), 50)
        combined = combine_bundles(bundle1, bundle1, 'rlap_keep', dist)
        assert_equal(combined, bundle1)
        # Test RLAP + closest
        combined = combine_bundles(bundle1, bundle2, 'rlap_closest', dist)
        assert_equal(len(combined), 50)
        combined = combine_bundles(bundle2, bundle1, 'rlap_closest', dist)
        assert_equal(len(combined), 50)
        combined = combine_bundles(bundle1, bundle1, 'rlap_keep', dist)
        assert_equal(combined, bundle1)
        # Test random pick
        combined = combine_bundles(bundle1, bundle2, 'random_pick', dist,
                                   n_stream=33)
        assert_equal(len(combined), 33)
        combined = combine_bundles(bundle2, bundle1, 'random_pick', dist,
                                   n_stream=33)
        assert_equal(len(combined), 33)


def test_compute_atlas_bundle():
    # Test argument errors
    assert_raises(TypeError, compute_atlas_bundle, in_dir=2)
    assert_raises(TypeError, compute_atlas_bundle, in_dir=in_dir.name,
                  mid_path=2)
    assert_raises(TypeError, compute_atlas_bundle, in_dir=in_dir.name,
                  n_point=2.78)
    assert_raises(ValueError, compute_atlas_bundle, in_dir='imaginary_folder')
    assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir.name,
                  out_dir='imaginary_folder')
    assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir.name,
                  n_stream_max=0)
    assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir.name,
                  n_stream_min=0)
    assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir.name,
                  n_point=0)

    # Test wrong mid_path
    assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir.name,
                  mid_path='fake_path')

    # Test bundle duplications
    df = pd.DataFrame(['AF_L', 'AF_L'], columns=['bundle'])
    fname = os.path.join(in_dir.name, 'bundles.tsv')
    df.to_csv(fname, sep='\t', index=False)
    assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir.name,
                  bundle_names=fname)

    # Test subject duplications
    df = pd.DataFrame({'participant': ['sub_1', 'sub_2', 'sub_2', 'sub_4']})
    fname = os.path.join(in_dir.name, 'participants.tsv')
    df.to_csv(fname, sep='\t', index=False)
    assert_raises(ValueError, compute_atlas_bundle, in_dir=in_dir.name,
                  subjects=fname)

    # Test full functionality
    out_dir = TemporaryDirectory()
    atlas, atlas_merged = compute_atlas_bundle(in_dir=in_dir.name,
                                               out_dir=out_dir.name,
                                               save_temp=False,
                                               merge_out=True)
    assert_equal(len(atlas), 3)
    assert_equal(len(atlas_merged), len(np.concatenate(atlas)))

    assert_equal(os.path.isfile(f'{out_dir.name}/AF_L.trk'), True)
    assert_equal(os.path.isfile(f'{out_dir.name}/CST_R.trk'), True)
    assert_equal(os.path.isfile(f'{out_dir.name}/CC_ForcepsMajor.trk'), True)
    assert_equal(os.path.isfile(f'{out_dir.name}/whole_brain.trk'), True)

    # Test specifying subject group + temporary files
    out_dir = TemporaryDirectory()
    df = pd.DataFrame({'participant': ['sub_1', 'sub_2', 'sub_3', 'sub_4'],
                       'group': ['control', 'patient', 'control', 'control']})
    fname = os.path.join(in_dir.name, 'participants.tsv')
    df.to_csv(fname, sep='\t', index=False)
    _ = compute_atlas_bundle(in_dir=in_dir.name, subjects=fname,
                             group='control', out_dir=out_dir.name,
                             save_temp=True)
    files = os.listdir(f'{out_dir.name}/temp/AF_L/step_0')
    trk_files = [file for file in files if file.endswith('.trk')]
    assert_equal(len(trk_files), 3)


if __name__ == '__main__':
    run_module_suite()
