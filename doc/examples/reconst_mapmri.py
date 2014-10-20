"""
================================================================
Continuous and analytical diffusion signal modelling with MAPMRI
================================================================

We show how to model the diffusion signal as a linear combination
of continuous functions from the MAPMRI basis [Ozarslan2013]_.
We also compute the analytical Orientation Distribution Function (ODF).

First import the necessary modules:
"""

from dipy.reconst.mapmri import MapmriModel
from dipy.viz import fvtk
from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere
from dipy.core.gradients import gradient_table

"""
Download and read the data for this tutorial.

fetch_isbi2013_2shell() provides data from the ISBI HARDI contest 2013 acquired 
for two shells at b-values 1500 and 2500.

The six parameters of these two functions define the ROI where to reconstruct
the data. They respectively correspond to (xmin,xmax,ymin,ymax,zmin,zmax)
with x, y, z and the three axis defining the spatial positions of the voxels.
"""

fetch_isbi2013_2shell()
img, gtab = read_isbi2013_2shell()
data = img.get_data()
data_small = data[10:40, 22, 10:40]

print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
data contains the voxel data and gtab contains a GradientTable
object (gradient information e.g. b-values). For example, to show the b-values
it is possible to write print(gtab.bvals).

Instantiate the MAPMRI Model.

radial_order is the radial order of the MAPMRI basis.

For details regarding the parameters see [Ozarslan2013]_.
"""

radial_order = 4
map_model = MapmriModel(gtab, radial_order=radial_order)

"""
Fit the MAPMRI model to the data
"""

mapfit = map_model.fit(data_small)

"""
Load an odf reconstruction sphere
"""

sphere = get_sphere('symmetric724')

"""
Compute the ODFs
"""

odf = mapfit.odf(sphere)
print('odf.shape (%d, %d, %d)' % odf.shape)

"""
Display the ODFs
"""

r = fvtk.ren()
sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, colormap='jet')
sfu.RotateX(-90)
fvtk.add(r, sfu)
fvtk.record(r, n_frames=1, out_path='odfs.png', size=(600, 600))

"""
.. figure:: odfs.png
   :align: center

   **Orientation distribution functions**.
   
.. [Ozarslan2013] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
               diffusion imaging method for mapping tissue microstructure",
               NeuroImage, 2013.

.. include:: ../links_names.inc

"""
