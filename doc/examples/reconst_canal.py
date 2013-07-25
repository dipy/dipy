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

#data, affine, gtab = two_shells_voxels(10, 40, 10, 40, 25, 26)
fetch_isbi2013_2shell()
data, gtab=read_isbi2013_2shell()
#data, affine, gtab = three_shells_voxels(45, 65, 35, 65, 33, 34)

print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
data contains the voxel data and gtab contains a GradientTable
object (gradient information e.g. b-values). For example to read the b-values
it is possible to write print(gtab.bvals).

Instantiate the SHORE Model.
"""

asm = ShoreModel(gtab)

"""
Fit the SHORE model to the data
"""

asmfit = asm.fit(data)


"""
Estimate the SHORE coefficient using the least square estimation with a l2 regularization.

radialOrder is the radial order of the SHORE basis.

zeta is the scale factor of the SHORE basis.

lambdaN and lambdaN are the radial and angular regularization constants, respectively.

For details regarding these four parameters see (Cheng J. et al, MICCAI workshop 2011) and 
(Merlet S. et al, Medical Image Analysis 2013).
"""

radialOrder = 4
zeta = 700
lambdaN=1e-8
lambdaL=1e-8
Cshore = asmfit.l2estimation(radialOrder=radialOrder, zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)


"""
Compute the ODF Spherical Harmonic coefficients  
"""

Csh = asmfit.odf()
print('Csh.shape (%d, %d, %d, %d)' % Csh.shape)


"""
Load an odf reconstruction sphere
"""

sphere = get_sphere('symmetric724')


"""
Evaluate the ODF in the direction provided by 'sphere'

sh_order is Spherical Harmonic order
"""

sh_order = radialOrder
odf = sh_to_sf(Csh, sphere, sh_order, basis_type="fibernav")
print('odf.shape (%d, %d, %d, %d)' % odf.shape)


"""
Display the ODFs
"""

r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(odf, sphere, colormap='jet'))
fvtk.show(r)