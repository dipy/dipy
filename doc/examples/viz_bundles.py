"""
=======================================
Visualize bundle and metrics on bundles
=======================================

"""

import numpy as np
import nibabel as nib
from dipy.viz import window, actor
from dipy.data import fetch_bundles_2_subjects, read_bundles_2_subjects

fetch_bundles_2_subjects()
dix = read_bundles_2_subjects()

fa = dix['fa']
affine = dix['affine']
bundle = dix['cst.right']

renderer = window.Renderer()

stream_actor = actor.line(bundle)

renderer.add(stream_actor)

window.show(renderer)

renderer.clear()

stream_actor2 = actor.line(bundle, fa)

renderer.add(stream_actor2)

window.show(renderer)

1/0

"""
Load colormap (FA map for this example)
"""
fa_file = nib.load(dname + "../fa_1x1x1.nii.gz")
fa_colormap = fa_file.get_data()
colormap_affine = fa_file.get_affine()

"""
4. Transform lines in the same coordinates
"""
transfo = np.linalg.inv(colormap_affine)
lines = [nib.affines.apply_affine(transfo, s) for s in lines]

"""
5. Generate and render fvtk streamline with scalar_bar
"""
width = 0.1
fvtk_tubes = vtk_a.streamtube(lines, fa_colormap, linewidth=width)
scalar_bar = vtk_a.scalar_bar(fvtk_tubes.GetMapper().GetLookupTable())

renderer = fvtk.ren()
fvtk.add(renderer, fvtk_tubes)
fvtk.add(renderer, scalar_bar)
fvtk.show(renderer)

"""
6. Generate and render fvtk streamline with scalar_bar
"""

saturation = [0.0, 1.0] # white to red
hue = [0.0, 0.0] # Red only

lut_cmap = vtk_a.colormap_lookup_table(hue_range=hue, saturation_range=saturation)

fvtk_tubes = vtk_a.streamtube(lines, fa_colormap, linewidth=width,
                              lookup_colormap=lut_cmap)

scalar_bar = vtk_a.scalar_bar(fvtk_tubes.GetMapper().GetLookupTable())

renderer = fvtk.ren()
fvtk.add(renderer, fvtk_tubes)
fvtk.add(renderer, scalar_bar)
fvtk.show(renderer)


