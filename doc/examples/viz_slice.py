
"""
==========================
Simple image visualization
==========================
"""
import numpy as np
import nibabel as nib
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell
from dipy.viz import window, actor
from dipy.core.histeq import histeq
from dipy.align.reslice import reslice

#fetch_sherbrooke_3shell()
#img, gtab = read_sherbrooke_3shell()


# fraw = '/home/eleftherios/Data/MPI_Elef/fa_1x1x1.nii.gz'
fraw = '/home/eleftherios/Data/Jorge_Rudas/tensor_fa.nii.gz'
img = nib.load(fraw)

# import nibabel as nib



data = img.get_data()
affine = img.get_affine()
zooms = img.get_header().get_zooms()[:3]
new_zooms = (2, 2, 2.)

print(affine)
print(data.shape)

data, affine = reslice(data, affine, zooms, new_zooms)

print(affine)
print(data.shape)

renderer = window.Renderer()

# S0 = data[..., 0]

vol = histeq(data)

world_coord = True
if world_coord:
    slice_actor = actor.slice(vol, affine)
else:
    slice_actor = actor.slice(vol)

renderer.add(slice_actor)

slice_actor2 = slice_actor.copy()

# slice_actor2.display_extent(64, 64, 0, 127, 0, 59)
# slice_actor2.display_extent(186/2, 186/2, 0, 231, 0, 185)
# slice_actor2.display_extent(120/2, 120/2, 0, 119, 0, 74)
slice_actor2.display_extent(data.shape[0]/2, data.shape[0]/2,
                            0, data.shape[1] - 1, 0, data.shape[2] - 1)

renderer.background((1, 1, 1))

renderer.add(slice_actor2)

window.show(renderer)
