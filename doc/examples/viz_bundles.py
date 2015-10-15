"""
========================================
Visualize bundles and metrics on bundles
========================================
"""

import numpy as np
from dipy.viz import window, actor
from dipy.data import fetch_bundles_2_subjects, read_bundles_2_subjects
from dipy.tracking.streamline import transform_streamlines

fetch_bundles_2_subjects()
dix = read_bundles_2_subjects(subj_id='subj_1', metrics=['fa'],
                              bundles=['cg.left', 'cst.right'])

fa = dix['fa']
affine = dix['affine']
bundle = dix['cg.left']

renderer = window.Renderer()

stream_actor = actor.line(bundle)

renderer.add(stream_actor)

window.show(renderer)

renderer.clear()

bundle_img = transform_streamlines(bundle, np.linalg.inv(affine))

stream_actor2 = actor.line(bundle_img, fa)

renderer.add(stream_actor2)

window.show(renderer)
renderer.clear()

stream_actor3 = actor.streamtube(bundle_img, fa, linewidth=0.1)
bar = actor.scalar_bar()

renderer.add(stream_actor3)
renderer.add(bar)

window.show(renderer)

renderer.clear()

hue = [0.0, 0.0]  # red only
saturation = [0.0, 1.0]  # white to red

lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                       saturation_range=saturation)

stream_actor4 = actor.streamtube(bundle_img, fa, linewidth=0.1,
                                 lookup_colormap=lut_cmap)
bar2 = actor.scalar_bar(lut_cmap)

renderer.add(stream_actor4)
renderer.add(bar2)

window.show(renderer)
