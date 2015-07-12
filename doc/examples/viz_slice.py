
"""
============================
Simple volume visualization
============================

Here we present an example

"""
import os
import nibabel as nib
from dipy.data import fetch_bundles_2_subjects
from dipy.viz import window, actor


fetch_bundles_2_subjects()

fname = os.path.join(os.path.expanduser('~'), '.dipy', 'exp_bundles_and_maps',
                     'bundles_2_subjects', 'subj_1', 't1_warped.nii.gz')

img = nib.load(fname)
data = img.get_data()
affine = img.get_affine()

renderer = window.Renderer()

mean, std = data[data > 0].mean(), data[data > 0].std()

value_range = (mean - 0.5 * std, mean + 1.5 * std)

world_coord = True

if world_coord:
    slice_actor = actor.slice(data, affine, value_range)
else:
    slice_actor = actor.slice(data, value_range=value_range)

renderer.add(slice_actor)

slice_actor2 = slice_actor.copy()

slice_actor2.display(slice_actor2.shape[0]/2, None, None)

renderer.background((1, 1, 1))

renderer.add(slice_actor2)

window.show(renderer, size=(600, 600))
