""" 

==========================
Lets make some streamlines
==========================

This example shows how to generate streamlines with Dipy_. First we will reconstruct
the datasets using a single Tensor model and next will use a Constant Solid
Angle ODF (Q-Ball) model. EuDX is the algorithm that we will use for fiber tracking.

First import the necessary modules:

``numpy`` is for numerical computation

"""

import numpy as np

"""
``nibabel`` is for loading imaging datasets
"""

import nibabel as nib

"""
``dipy.reconst`` is for the reconstruction algorithms which we use to create voxel models from the raw data. 
"""

from dipy.reconst.dti import TensorModel

"""
``dipy.tracking`` is for fiber tracking.
"""

from dipy.tracking.eudx import EuDX

"""
``dipy.data`` is used for small datasets that we use in tests and examples.
"""

from dipy.data import fetch_beijing_dti


"""
Fetch will download the raw dMRI dataset of a single subject. The size of the dataset is 51 MBytes.
You only need to fetch once.
"""

fetch_beijing_dti()

"""
Next, we read the saved dataset
"""

from dipy.data import read_beijing_dti

img, gtab = read_beijing_dti()

"""
img contains a nibabel Nifti1Image object (data) and gtab contains a GradientTable
object (gradient information e.g. b-values).
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
This dataset has anisotropic voxel sizes, therefore reslicing is necessary
"""

affine = img.get_affine()

"""
Load and show the zooms which hold the voxel size.
"""

zooms = img.get_header().get_zooms()[:3]

"""
The voxel size here is ``(1.79, 1.79, 2.5)``.

We now set the required new voxel size.
"""

new_zooms = (2., 2., 2.)

"""
Which is ``(2.0, 2.0, 2.0)``

Start reslicing. Trilinear interpolation is used by default.
"""

from dipy.align.aniso2iso import resample

data2, affine2 = resample(data, affine, zooms, new_zooms)

print('data2.shape (%d, %d, %d, %d)' % data2.shape)

"""
Lets also create a simple mask.
"""

mask = data2[..., 0] > 50

"""
Now that we have prepared the datasets we can go forward with the voxel
reconstruction. First, we instantiate the Tensor model in the following way.
"""

tenmodel = TensorModel(gtab)

"""
Fitting the data is very simple. We just need to calling the fit method of the
TensorModel in the followin way:
"""

tenfit = tenmodel.fit(data2, mask)

"""
The fit method creates a TensorFit object which contains the fitting parameters
and other attributes of the model. For example we can generate fractional
anisotropy.
"""

FA = tenfit.fa

"""
In the background of the image the fitting will not be accurate there is no
signal and possibly we will find FA values with nans (not a number). We can
easily remove these in the following way.
"""

FA[np.isnan(FA)] = 0

"""
EuDX takes as input discretized voxel directions on a unit sphere. Therefore,
it is necessary to discretize the eigen vectors before feeding them in EuDX.

For the discretization procedure we use an evenly distributed sphere of 724
points which we can access using the get_sphere function.
"""

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

"""
We use quantize_evecs (evecs here stands for eigen vectors) to apply the
discretization.
"""

from dipy.reconst.dti import quantize_evecs

peak_indices = quantize_evecs(tenfit.evecs, sphere.vertices)

"""
EuDX is the fiber tracking algorithm that we use in this example.
The most important parameters are the first one which represents the
magnitude of the peak of a scalar anisotropic function, the 
second which represents the indices of the discretized directions of 
the peaks and odf_vertices are the vertices of the input sphere.
"""

from dipy.tracking.eudx import EuDX

eu = EuDX(FA, peak_indices, odf_vertices = sphere.vertices, a_low=0.2)

tensor_streamlines = [streamline for streamline in eu]

"""
We use here a python vtk module to show the streamlines
"""

from dipy.viz import fvtk

"""
Create a scene
"""

r=fvtk.ren()

"""
Every streamline will be coloured according to its orientation
"""

from dipy.viz.colormap import line_colors

"""
fvtk.line adds a streamline actor for streamline visualization
and fvtk.add adds this actor in the scene
"""

fvtk.add(r, fvtk.line(tensor_streamlines, line_colors(tensor_streamlines)))

print('Saving illustration as tensor_tracks.png')
fvtk.record(r, n_frames=1, out_path='tensor_tracking.png', size=(600, 600))

"""
Lets now repeat the same procedure with a CSA ODF model
"""

from dipy.reconst.shm import CsaOdfModel, normalize_data

#data2 = data2.astype('f8')

"""
We instantiate our CSA model with sperical harmonic order of 4
"""

csamodel = CsaOdfModel(gtab, 4)

"""
Peaks from model is used to calculate properties of the ODFs and return for
example the peaks and their indices, or GFA which is similar to FA but for ODF
based models.
"""

from dipy.reconst.odf import peaks_from_model

peaks = peaks_from_model(model=csamodel, 
                         data=data2, 
                         sphere=sphere, 
                         relative_peak_threshold=.8, 
                         min_separation_angle=45, 
                         mask=mask, 
                         normalize_peaks=True)

"""
This time we will not use FA as input to EuDX but we will use directly
the maximum peaks of the ODF. The a_low threshold is the 
"""

eu = EuDX(peaks.peak_values, 
          peaks.peak_indices, 
          odf_vertices = sphere.vertices, 
          a_low=0.2)

csa_streamlines = [streamline for streamline in eu]

"""
Clear the scene
"""

fvtk.clear(r)

fvtk.add(r, fvtk.line(csa_streamlines, line_colors(csa_streamlines)))

fvtk.record(r, n_frames=1, out_path='csa_tracking.png', size=(600, 600))

"""
In the previous example we show that in the Tensor case we discretized the
eigen_vectors and for the CSA case we used peaks_from_model as the CSA is a
model that creates ODFs. However, the Tensor creates ODFs too which are always
ellipsoids. So, we could have used the Tensor with peaks_from_model.
"""

"""
We can now save the results in the disk. For this purpose we can use the
TrackVis format (*.trk). First, we need to create a header.
"""

hdr = nib.trackvis.empty_header()
hdr['voxel_size'] = new_zooms
hdr['voxel_order'] = 'LAS'
hdr['dim'] = data2.shape[:3]

"""
Then we need to fill the streamlines.
"""

tensor_streamlines = ((sl, None, None) for sl in tensor_streamlines)

ten_sl_fname = 'tensor_streamline.trk'

"""
Save the streamlines.
"""

nib.trackvis.write(ten_sl_fname, tensor_streamlines, hdr)

"""
Lets repeat the same steps for the csa_streamlines.
"""

csa_streamlines = ((sl, None, None) for sl in csa_streamlines)

csa_sl_fname = 'csa_streamline.trk'

nib.trackvis.write(csa_sl_fname, csa_streamlines, hdr)


"""
.. include:: ../links_names.inc

"""



