import numpy as np
# import nibabel as nib
from dipy.reconst.opt_msd import MultiShellResponse
from dipy.reconst.csdeconv import auto_response
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
from dipy.sims.voxel import (single_tensor)
from dipy.reconst import shm
from dipy.data import default_sphere
from dipy.core.gradients import GradientTable
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
import dipy.reconst.dti as dti
from dipy.reconst.opt_msd import MultiShellDeconvModel
from dipy.viz import window, actor
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')


def show_slicer_and_pick(data, affine):
    from dipy.viz import window, actor, ui

    im_actor = actor.slicer(data, affine)
    shape = data.shape

    renderer = window.Renderer()
    renderer.projection('parallel')
    show_m = window.ShowManager(renderer, size=(1200, 900))
    show_m.initialize()
    line_slider_z = ui.LineSlider2D(min_value=0,
                                    max_value=shape[2] - 1,
                                    initial_value=shape[2] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    def change_slice_z(slider):
        z = int(np.round(slider.value))
        im_actor.display_extent(0, shape[0] - 1, 0, shape[1] - 1, z, z)
        show_m.ren.reset_camera()
        show_m.ren.reset_clipping_range()
        show_m.render()

    line_slider_z.on_change = change_slice_z

    label_position = ui.TextBlock2D(text='Position:')
    label_value = ui.TextBlock2D(text='Value:')
    result_position = ui.TextBlock2D(text='')
    result_value = ui.TextBlock2D(text='')

    panel_picking = ui.Panel2D(size=(250, 225),
                               position=(20, 20),
                               color=(0, 0, 0),
                               opacity=0.75,
                               align="left")

    panel_picking.add_element(label_position, (0.1, 0.55))
    panel_picking.add_element(label_value, (0.1, 0.25))
    panel_picking.add_element(result_position, (0.45, 0.55))
    panel_picking.add_element(result_value, (0.45, 0.25))
    panel_picking.add_element(line_slider_z, (0.1, 0.75))
    show_m.ren.add(panel_picking)

    def left_click_callback(obj, ev):
        """Get the value of the clicked voxel and show it in the panel."""
        event_pos = show_m.iren.GetEventPosition()

        obj.picker.Pick(event_pos[0],
                        event_pos[1],
                        0,
                        show_m.ren)

        i, j, k = obj.picker.GetPointIJK()
        result_position.message = '({}, {}, {})'.format(str(i), str(j), str(k))
        result_value.message = '%.8f' % data[i, j, k]

    im_actor.SetInterpolate(False)
    im_actor.AddObserver('LeftButtonPressEvent', left_click_callback, 1.0)

    show_m.ren.add(im_actor)
    show_m.ren.reset_camera()
    show_m.ren.reset_clipping_range()
    show_m.render()
    show_m.start()


# static file paths for experiments
fbvals = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fd0.neuro-dwi/dwi.bvals'
fbvecs = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fd0.neuro-dwi/dwi.bvecs'
fdwi = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fd0.neuro-dwi/dwi.nii.gz'
ft1 = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fcf.neuro-anat-t1w.acpc_aligned/t1.nii.gz'

t1, t1_affine = load_nifti(ft1)

dwi, dwi_affine = load_nifti(fdwi)
b0_mask, mask = median_otsu(dwi)

# t1[mask == 0] = 0

print("Data Loaded!")

"""
Now we will define the other two parameters for the segmentation algorithm.
We will segment three classes, namely corticospinal fluid (CSF), white matter
(WM) and gray matter (GM).
"""
nclass = 3
"""
Then, the smoothness factor of the segmentation. Good performance is achieved
with values between 0 and 0.5.
"""
beta = 0.1

# denoising
sigma = estimate_sigma(t1, True, N=4)

t1[mask == 0] = 0

t1_den = nlmeans(t1, sigma=sigma)

# save_nifti('t1_masked.nii.gz', t1, t1_affine)
# save_nifti('t1_class_masked.nii.gz', PVE, t1_affine)

bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs)

# fitting the model with DTI
tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(dwi, mask)

# save_nifti('t1_denoised.nii.gz', t1_den, t1_affine)

# getting the mean diffusivities and FAs from DTI
FA = tenfit.fa
MD = tenfit.md

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(t1_den, nclass,
                                                              beta)

csf = PVE[..., 0]
cgm = PVE[..., 1]

save_nifti('pve.nii.gz', PVE, dwi_affine)

indices_csf = np.where(((FA < 0.2) & (csf > 0.95)))
indices_cgm = np.where(((FA < 0.2) & (cgm > 0.95)))

selected_csf = np.zeros(FA.shape, dtype='bool')
selected_cgm = np.zeros(FA.shape, dtype='bool')

selected_csf[indices_csf] = True
selected_cgm[indices_cgm] = True

csf_md = np.mean(tenfit.md[selected_csf])
cgm_md = np.mean(tenfit.md[selected_cgm])

# save_nifti('selected_csf.nii.gz', selected_csf, dwi_affine)
# save_nifti('selected_cgm.nii.gz', selected_cgm, dwi_affine)

save_nifti('md.nii.gz', MD, dwi_affine)
save_nifti('fa.nii.gz', FA, dwi_affine)

# center = np.zeros(FA.shape)
# center[63 - 10: 63 + 10, 113 - 10: 113 + 10, 75 - 10: 75 + 10] = 1
# save_nifti('center.nii.gz', center, dwi_affine)

# generating the autoresponse
dwi[mask == 0] = 0
response, ratio = auto_response(gtab, dwi, roi_radius=10, fa_thr=0.7)
evals_d = response[0]


def sim_response(sh_order=8, bvals=bvals, evals=evals_d, csf_md=csf_md,
                 gm_md=cgm_md):
    bvals = np.array(bvals, copy=True)
    evecs = np.zeros((3, 3))
    z = np.array([0, 0, 1.])
    evecs[:, 0] = z
    evecs[:2, 1:] = np.eye(2)

    n = np.arange(0, sh_order + 1, 2)
    m = np.zeros_like(n)

    big_sphere = default_sphere.subdivide()
    theta, phi = big_sphere.theta, big_sphere.phi

    B = shm.real_sph_harm(m, n, theta[:, None], phi[:, None])
    A = shm.real_sph_harm(0, 0, 0, 0)

    response = np.empty([len(bvals), len(n) + 2])
    for i, bvalue in enumerate(bvals):
        gtab = GradientTable(big_sphere.vertices * bvalue)
        wm_response = single_tensor(gtab, 1., evals, evecs, snr=None)
        response[i, 2:] = np.linalg.lstsq(B, wm_response)[0]

        response[i, 0] = np.exp(-bvalue * csf_md) / A
        response[i, 1] = np.exp(-bvalue * gm_md) / A

    return MultiShellResponse(response, sh_order, bvals)


response_msd = sim_response(sh_order=4, bvals=bvals, evals=evals_d,
                            csf_md=csf_md, gm_md=cgm_md)

msd_model = MultiShellDeconvModel(gtab, response_msd)

# data = dwi[63 - 10: 63 + 10, 113 - 10: 113 + 10, 75: 75 + 10]
# data = dwi[63 - 10: 63 + 10, 113 - 10: 113 + 10, 75: 75 + 10]

data = dwi[:, :, 68: 68 + 1]

# odf_mask[63 - 10: 63 + 10, 113 - 10: 113 + 10, 75: 75 + 1] = 1

msd_fit = msd_model.fit(data)
msd_odf = msd_fit.odf(sphere)
fodf_spheres = actor.odf_slicer(msd_odf, sphere=sphere, scale=0.9, norm=True,
                                colormap='plasma')
interactive = True
ren = window.Renderer()
ren.add(fodf_spheres)

print('Saving illustration as msd_odfs.png')
window.record(ren, out_path='msd_odfs.png', size=(600, 600))
if interactive:
    window.show(ren)
