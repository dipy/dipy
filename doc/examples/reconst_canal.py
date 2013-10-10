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
from dipy.reconst.canal import ShoreModel
from dipy.reconst.shm import sh_to_sf
from dipy.viz import fvtk
from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere
from dipy.core.gradients import gradient_table

"""
Download and read the data for this tutorial.

two_shells_voxels() provides data from the ISBI HARDI contest 2013 acquired 
for two shells at b-values 1500 and 2500.

three_shells_voxels() provides a humain brain data acquired for for three 
shells at b-values 1000 and 2000 and 3500.

The six parameters of these two functions define the ROI where to reconstruct
the data. They respectively correspond to (xmin,xmax,ymin,ymax,zmin,zmax)
with x,y and the three axis defining the spatial positions of the voxels.
"""

fetch_isbi2013_2shell()
img, gtab=read_isbi2013_2shell()
data = img.get_data()
data_small=data[10:40,10:40,25]

print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
data contains the voxel data and gtab contains a GradientTable
object (gradient information e.g. b-values). For example to read the b-values
it is possible to write print(gtab.bvals).

Instantiate the SHORE Model.

radialOrder is the radial order of the SHORE basis.

zeta is the scale factor of the SHORE basis.

lambdaN and lambdaN are the radial and angular regularization constants, respectively.

For details regarding these four parameters see (Cheng J. et al, MICCAI workshop 2011) and 
(Merlet S. et al, Medical Image Analysis 2013).
"""

radialOrder = 6
zeta = 700
lambdaN=1e-8
lambdaL=1e-8
asm = ShoreModel(gtab, radialOrder=radialOrder, zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)

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