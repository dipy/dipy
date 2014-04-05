"""
==========================================
Symmetric Diffeomorphic Registration in 3D
==========================================
This example explains how to register 3D volumes using the Symmetric Normalization 
(SyN) algorithm proposed by Avants et al. [citation needed] (also implemented in
the ANTS software [citation needed])

We'll register two 3D volumes from different modalities (FA computed from
diffusion MRI and T1) using SyN with the Cross Correlation (CC) metric.
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
import os.path

fname_static = '/opt/registration/data/RGB_T1_ROI/1.5T/Maxime.Gagnon/t1_nlm.nii.gz'
fname_moving = '/opt/registration/data/RGB_T1_ROI/1.5T/Maxime.Gagnon/fa.nii.gz'

nib_static = nib.load(fname_static)
nib_moving = nib.load(fname_moving)

"""
First we need an initial affine registration that brings the moving volume to
the static's domain
"""

def affine_registration(static, moving):
    from nipy.io.files import nipy2nifti, nifti2nipy
    from nipy.algorithms.registration import HistogramRegistration, resample
    nipy_static = nifti2nipy(static)
    nipy_moving = nifti2nipy(moving)
    similarity = 'crl1' #'crl1' 'cc', 'mi', 'nmi', 'cr', 'slr'
    interp = 'tri' #'pv', 'tri',
    renormalize = True
    optimizer = 'powell'
    R = HistogramRegistration(nipy_static, nipy_moving, similarity=similarity,
                          interp=interp, renormalize=renormalize)
    T = R.optimize('affine', optimizer=optimizer)
    warped= resample(nipy_moving, T, reference=nipy_static, interp_order=1)
    warped = nipy2nifti(warped, strict=True)
    return warped, T

warped, affine_init = affine_registration(nib_static, nib_moving)
static = nib_static.get_data().squeeze().astype(np.float32)
moving = warped.get_data().squeeze().astype(np.float32)

def renormalize_image(image):
    m=np.min(image)
    M=np.max(image)
    if(M-m<1e-8):
        return image
    return 127.0*(image-m)/(M-m)


def plot_middle_slices(V, fname=None):
    sh=V.shape
    axial = renormalize_image(V[sh[0]//2, :, :]).astype(np.int8)
    coronal = renormalize_image(V[:, sh[1]//2, :]).astype(np.int8)
    sagital = renormalize_image(V[:, :, sh[2]//2]).astype(np.int8)

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(axial, cmap = plt.cm.gray)
    plt.title('Axial')
    plt.subplot(1,3,2)
    plt.imshow(coronal, cmap = plt.cm.gray)
    plt.title('Coronal')
    plt.subplot(1,3,3)
    plt.imshow(sagital, cmap = plt.cm.gray)
    plt.title('Sagital')
    if fname is not None:
        from time import sleep
        sleep(1)
        plt.savefig(fname, bbox_inches='tight')


def plot_middle_slices_coronal(L, R, ltitle='Left', rtitle='Right', fname=None):
    sh=L.shape

    colorImage=np.zeros(shape=(sh[0], sh[2], 3), dtype=np.int8)
    ll=renormalize_image(L[:,sh[1]//2,:]).astype(np.int8)
    rr=renormalize_image(R[:,sh[1]//2,:]).astype(np.int8)
    colorImage[...,0]=ll*(ll>ll[0,0])
    colorImage[...,1]=rr*(rr>rr[0,0])

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(ll, cmap = plt.cm.gray)
    plt.title(ltitle)
    plt.subplot(1,3,2)
    plt.imshow(colorImage)
    plt.title('Overlay')
    plt.subplot(1,3,3)
    plt.imshow(rr, cmap = plt.cm.gray)
    plt.title(rtitle)
    if fname is not None:
        from time import sleep
        sleep(1)
        plt.savefig(fname, bbox_inches='tight')

plot_middle_slices_coronal(static,moving, 'Static', 'Moving', 'input_coronal.png')

"""
.. figure:: input_coronal.png
   :align: center

**Pre-aligned input images. Left: static (T1). Right: moving (FA)**.
"""


forward_fname = 'forward_field.npy'
backward_fname = 'backward_field.npy'


if os.path.isfile(forward_fname) and os.path.isfile(backward_fname):
    print("Diffeomorphic maps found [forward:%s], [backward:%s]."%(forward_fname, backward_fname))
    print("Optimization skipped.")
    forward = np.load(forward_fname)
    backward = np.load(backward_fname)
    mapping = DiffeomorphicMap(3, forward, backward, None, None)
else:

    """
    We want to find an invertible map that transforms the moving image into the
    static image. Let's use the Cross Correlation metric, since it works well
    for monomodal and some multi-modal registration tasks.
    """

    metric = CCMetric(3)

    """
    Now we define an instance of the optimizer of the metric. The SyN algorithm uses
    a multi-resolution approach by building a Gaussian Pyramid. We instruct the
    optimizer to perform at most [n_0, n_1, ..., n_k] iterations at each level of
    the pyramid. The 0-th level corresponds to the finest resolution.  
    """

    opt_iter = [5, 10, 10]
    registration_optimizer = SymmetricDiffeomorphicRegistration(metric, opt_iter)
    registration_optimizer.verbosity = 2

    """
    Execute the optimization, which returns a DiffeomorphicMap object,
    that can be used to register images back and forth between the static and moving
    domains
    """

    mapping = registration_optimizer.optimize(static, moving, None)

    np.save(forward_fname, mapping.forward)
    np.save(backward_fname, mapping.backward)

"""
It is a good idea to visualize the resulting deformation map to make sure the
result is reasonable (visually, at least). You can visualize the whole 3D grid 
using your favorite visualization tool (e.g. the fiber navigator)
"""

def draw_lattice_3d(dims, delta=10):
    dims=np.array(dims)
    nsquares=(dims-1)/(delta+1)
    lattice=np.zeros(shape=dims[:3], dtype=np.float32)
    lattice[...]=127
    for i in range(nsquares[0]+1):
        lattice[i*(delta+1), :, :]=0
    for j in range(nsquares[1]+1):
        lattice[:, j*(delta+1), :]=0
    for k in range(nsquares[2]+1):
        lattice[:, :, k*(delta+1)]=0
    return lattice

def save_deformed_lattice_3d(mapping, apply_inverse, oname, img_fname=None):
    r"""
    Plots a regular grid deformed under the action of the given mapping.

    Parameters
    ----------
    mapping : DiffeomorphicMap object
        the mapping acting on the regular grid
    apply_inverse : boolean
        if True, plots grid under the action of the inverse mapping, else
        plots the action of the direct mapping
    oname : string
        the file name to be used to store the resulting deformed lattice 
        (in nifti format)
    """
    if apply_inverse:
        shape = mapping.forward.shape
        grid=draw_lattice_3d(shape)
        warped=mapping.transform_inverse(grid).astype(np.int16)
    else:
        shape = mapping.backward.shape
        grid=draw_lattice_3d(shape)
        warped=mapping.transform(grid).astype(np.int16)
    
    img=nib.Nifti1Image(warped, np.eye(4))
    img.to_filename(oname)
    if img_fname is not None:
        from time import sleep
        plot_middle_slices(warped)
        sleep(1)
        plt.savefig(img_fname, bbox_inches='tight')


save_deformed_lattice_3d(mapping, False, 'deformed_lattice.nii.gz', 'def_grid_coronal.png')

"""
.. figure:: def_grid_coronal.png
   :align: center

**Deformed grid under the action of the (direct) mapping**.
"""

save_deformed_lattice_3d(mapping, True, 'inv_deformed_lattice.nii.gz', 'invdef_grid_coronal.png')

"""
.. figure:: invdef_grid_coronal.png
   :align: center

**Deformed grid under the action of the (inverse) mapping**.
"""

"""
Now let's warp the moving image and see if it gets similar to the static image
"""

warped_moving = mapping.transform(moving)

"""
To visually check the overlap of the static image with the transformed moving
image, we can plot them on top of each other with different channels to see
where the differences are located
"""

plot_middle_slices_coronal(static, warped_moving, 'Static', 'Warped moving', 'warped_moving.png')

"""
.. figure:: warped_moving.png
    :align: center

**Moving image transformed under the (direct) transformation in green
on top of the static image (in red)**.
"""

"""
And we can also apply the inverse mapping to verify that the warped static image
is similar to the moving image 
"""

warped_static = mapping.transform_inverse(static)

plot_middle_slices_coronal(warped_static, moving, 'Warped static', 'Moving', 'warped_static.png')

"""
.. figure:: warped_static.png
    :align: center

**Static image transformed under the (inverse) transformation in red
on top of the moving image (in green)**.
"""
