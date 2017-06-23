# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:04:25 2017

@author: elef
"""
import numpy as np
import nibabel as nib
from os.path import expanduser, join
import vtk
from dipy.viz import actor, window, widget

"""Here define a histogram normalization to adjust intensity
"""
def histogram_normalization(data, Rate):

    g, h = np.histogram(data)
    low = data.min()
    high = data.max()
    range_intensity = high - low
    bound = (range_intensity * Rate) + low

    return bound


def show_volume(vol, vol2, rate, normalization=True, affine=np.eye(4), opacity=1.):

    ren = window.Renderer()
#    rate = 0.38
    high = histogram_normalization(vol, rate)
    #high = histogram_normalization(vol, rate)

    shape = vol.shape

    image_actor = actor.slicer(vol, affine)
    if normalization:
        image_actor2 = actor.slicer(vol2, affine, value_range=(0, high))
        image_actor = actor.slicer(vol, affine, value_range=(0, high))
    else:
        image_actor2 = actor.slicer(vol2, affine)

    slicer_opacity = opacity  # .6
    image_actor.opacity(slicer_opacity)
    image_actor2.opacity(slicer_opacity)

    ren.add(image_actor)
    ren.add(image_actor2)

    image_actor.SetPosition(300,0,0)
    image_actor2.SetPosition(0,0,0)

    show_m = window.ShowManager(ren, size=(800, 700))
    show_m.picker = vtk.vtkCellPicker()
    show_m.picker.SetTolerance(0.002)

    show_m.initialize()
    show_m.iren.SetPicker(show_m.picker)
    show_m.picker.Pick(10, 10, 0, ren)

    def change_slice(obj, event):
        z = int(np.round(obj.get_value()))
        image_actor.display_extent(0, shape[0] - 1,
                                   0, shape[1] - 1, z, z)
        image_actor2.display_extent(0, shape[0] - 1,
                                   0, shape[1] - 1, z, z)
        ren.reset_clipping_range()

    slider = widget.slider(show_m.iren, show_m.ren,
                           callback=change_slice,
                           min_value=0,
                           max_value=shape[2] - 1,
                           value=shape[2] / 2,
                           label="Move slice",
                           right_normalized_pos=(.98, 0.6),
                           size=(120, 0), label_format="%0.lf",
                           color=(1., 1., 1.),
                           selected_color=(0.86, 0.33, 1.))

    global size
    size = ren.GetSize()

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():

            slider.place(ren)
            size = obj.GetSize()

    def annotate_pick(obj, event):
        I, J, K = obj.GetCellIJK()

        print('Value of voxel [%i, %i, %i]=%s' % (I, J, K, str(np.round(vol[I, J, K]))))
        # print("Picking 3d position")
        # print(obj.GetPickPosition())

    show_m.picker.AddObserver("EndPickEvent", annotate_pick)
    show_m.initialize()
    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()

'''
home = expanduser('~')
dname_template = join(home, 'Data1', 'mni_icbm152_nlin_asym_09c')

ft1 = join(dname_template, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')
fgm = join(dname_template, 'mni_icbm152_gm_tal_nlin_asym_09c.nii')
fwm = join(dname_template, 'mni_icbm152_wm_tal_nlin_asym_09c.nii')
fcsf = join(dname_template, 'mni_icbm152_csf_tal_nlin_asym_09c.nii')
fmask = join(dname_template, 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')


ft1 = "/Users/tiwanyan/Dropbox/IPMA/assignment1/data/T1_healthy.nii.gz"
t1t_img = nib.load(ft1)
t1t = t1t_img.get_data()

ft2 = "/Users/tiwanyan/Dropbox/IPMA/assignment1/data/T1_tumor.nii.gz"
# apply mask
t1t_mask_img = nib.load(fmask)
t1t_mask = t1t_mask_img.get_data()
t1t_unmask_img = nib.load(ft1)
t1t_unmask = t1t_unmask_img.get_data()
t1t[t1t_mask == 0] = 0


ft1_moving = join(home, 'tissue_data', 't1_brain.nii.gz')
t1m_img = nib.load(ft1_moving)
t1m = t1m_img.get_data()

"""Here is Goal 0, show both template and data
"""
t1m_img = nib.load(ft2)
t1m = t1m_img.get_data()
show_volume(t1t, t1m, 0.4)

"""Here is Goal 1, show the mask and unmask difference
"""
#show_volume(t1t,t1t_unmask,False)

#nib.save(nib.Nifti1Image(t1t, t1t_mask_img.affine), 't1t_brain_only.nii.gz')
'''
