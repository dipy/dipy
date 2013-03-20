"""
=================================================
Reconstruct with Constant Solid Angle (QBall)
=================================================

We show how to apply a Constant Solid Angle ODF (Q-Ball) model from Aganj et.
al (MRM 2010) to your datasets.

First import the necessary modules:
"""

import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, normalize_data
from dipy.reconst.odf import peaks_from_model

"""
Download and read the data for this tutorial.
"""

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

"""
img contains a nibabel Nifti1Image object (data) and gtab contains a GradientTable
object (gradient information e.g. b-values). For example to read the b-values
it is possible to write print(gtab.bvals).

Load the raw diffusion data and the affine.
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
data.shape ``(81, 106, 76, 160)``

Remove most of the background in the following simple way.
"""

mask = data[..., 0] > 50

"""
We instantiate our CSA model with sperical harmonic order of 4
"""

csamodel = CsaOdfModel(gtab, 4)

"""
`Peaks_from_model` is used to calculate properties of the ODFs (Orientation
Distribution Function) and return for
example the peaks and their indices, or GFA which is similar to FA but for ODF
based models. This function mainly needs a reconstruction model, the data and a
sphere as input. The sphere is an object that represents the spherical discrete
grid where the ODF values will be evaluated.
"""

sphere = get_sphere('symmetric724')

csapeaks = peaks_from_model(model=csamodel,
                            data=data,
                            sphere=sphere,
                            relative_peak_threshold=.8,
                            min_separation_angle=45,
                            mask=mask,
                            return_odf=False,
                            normalize_peaks=True)

GFA = csapeaks.gfa

print('GFA.shape (%d, %d, %d)' % GFA.shape)

"""
GFA.shape ``(81, 106, 76)``

Apart from GFA csapeaks has also the attributes peak_values, peak_indices and
ODF. peak_values shows the maxima values of the ODF and peak_indices gives us
their position on the discrete sphere that was used to do the reconstruction of
the ODF. In order to obtain the full ODF return_odf should be True. Before
enabling this option make sure that you have enough memory.

Finally lets try to visualize the orientation distribution functions of a small
rectangular area around the middle of our datasets.
"""

i,j,k,w = np.array(data.shape) / 2
data_small  = data[i-5:i+5, j-5:j+5, k-2:k+2]
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

from dipy.viz import fvtk
r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(csamodel.fit(data_small).odf(sphere),
							  sphere, colormap='jet'))
print('Saving illustration as csa_odfs.png')
fvtk.record(r, n_frames=1, out_path='csa_odfs.png', size=(600, 600))

"""
.. figure:: csa_odfs.png
   :align: center

   **Constant Solid Angle ODFs**.

.. include:: ../links_names.inc

"""
