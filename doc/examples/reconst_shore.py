"""
====================================================
Continuous and analytical diffusion signal modelling
====================================================

We show how to model the diffusion signal as a linear combination
of continuous functions from the SHORE basis (Ozarslan et al. ISMRM 2009).
We also compute analytically the ODF.

First import the necessary modules:
"""

#from dipy.data import three_shells_voxels, two_shells_voxels, get_sphere
from dipy.reconst.shore import ShoreModel
from dipy.reconst.shm import sh_to_sf
from dipy.viz import fvtk
from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere
from dipy.core.gradients import gradient_table

"""
Download and read the data for this tutorial.

two_shells_voxels() provides data from the ISBI HARDI contest 2013 acquired
for two shells at b-values 1500 and 2500.

The six parameters of these two functions define the ROI where to reconstruct
the data. They respectively correspond to (xmin,xmax,ymin,ymax,zmin,zmax)
with x, y, z and the three axis defining the spatial positions of the voxels.
"""

fetch_isbi2013_2shell()
img, gtab = read_isbi2013_2shell()
data = img.get_data()
data_small = data[10:40, 10:40, 25]

print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
data contains the voxel data and gtab contains a GradientTable
object (gradient information e.g. b-values). For example, to show the b-values
it is possible to write print(gtab.bvals).

Instantiate the SHORE Model.

radial_order is the radial order of the SHORE basis.

zeta is the scale factor of the SHORE basis.

lambdaN and lambdaN are the radial and angular regularization constants,
respectively.

For details regarding these four parameters see [Cheng2011]_ and [Merlet2013].
"""

radial_order = 6
zeta = 700
lambdaN = 1e-8
lambdaL = 1e-8
asm = ShoreModel(gtab, radial_order=radial_order,
                 zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)

"""
Fit the SHORE model to the data
"""

asmfit = asm.fit(data_small)

"""
Load an odf reconstruction sphere
"""

sphere = get_sphere('symmetric724')

"""
Compute the ODF
"""

odf = asmfit.odf(sphere)
print('odf.shape (%d, %d, %d)' % odf.shape)

"""
Display the ODFs
"""

r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(odf, sphere, colormap='jet'))
fvtk.show(r)
fvtk.record(r, n_frames=1, out_path='odfs.png', size=(600, 600))

"""
.. [Cheng2011] Cheng J. et. al , "Theoretical Analysis and Pratical Insights
			   on EAP Estimation via Unified HARDI Framework", MICCAI
			   workshop workshop on Computational Diffusion MRI, 2011.

.. [Merlet2013] Merlet S. et. al, "Continuous diffusion signal, EAP and ODF
           estimation via Compressive Sensing in diffusion MRI", Medical
           Image Analysis, 2013.

.. include:: ../links_names.inc

"""
