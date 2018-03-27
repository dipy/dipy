"""
==================================================================
Continuous and analytical diffusion signal modelling with 3D-SHORE
==================================================================

We show how to model the diffusion signal as a linear combination
of continuous functions from the SHORE basis [Merlet2013]_.
We also compute the analytical Orientation Distribution Function (ODF).

First import the necessary modules:
"""

from dipy.reconst.shore import ShoreModel
from dipy.viz import window, actor
from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere

"""
Download and read the data for this tutorial.

``fetch_isbi2013_2shell()`` provides data from the `ISBI HARDI contest 2013
<http://hardi.epfl.ch/static/events/2013_ISBI/>`_ acquired for two shells at
b-values 1500 $s/mm^2$ and 2500 $s/mm^2$.

The six parameters of these two functions define the ROI where to reconstruct
the data. They respectively correspond to ``(xmin,xmax,ymin,ymax,zmin,zmax)``
with x, y, z and the three axis defining the spatial positions of the voxels.
"""

fetch_isbi2013_2shell()
img, gtab = read_isbi2013_2shell()
data = img.get_data()
data_small = data[10:40, 22, 10:40]

print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
``data`` contains the voxel data and ``gtab`` contains a ``GradientTable``
object (gradient information e.g. b-values). For example, to show the b-values
it is possible to write::

    ``print(gtab.bvals)``

Instantiate the SHORE Model.

``radial_order`` is the radial order of the SHORE basis.

``zeta`` is the scale factor of the SHORE basis.

``lambdaN`` and ``lambdaL`` are the radial and angular regularization
constants, respectively.

For details regarding these four parameters see [Cheng2011]_ and [Merlet2013]_.
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
Compute the ODFs
"""

odf = asmfit.odf(sphere)
print('odf.shape (%d, %d, %d)' % odf.shape)

"""
Display the ODFs
"""

# Enables/disables interactive visualization
interactive = False

ren = window.Renderer()
sfu = actor.odf_slicer(odf[:, None, :], sphere=sphere, colormap='plasma', scale=0.5)
sfu.RotateX(-90)
sfu.display(y=0)
ren.add(sfu)
window.record(ren, out_path='odfs.png', size=(600, 600))
if interactive:
    window.show(ren)

"""
.. figure:: odfs.png
   :align: center

   Orientation distribution functions.

References
----------

.. [Merlet2013] Merlet S. et al., "Continuous diffusion signal, EAP and ODF
   estimation via Compressive Sensing in diffusion MRI", Medical Image
   Analysis, 2013.

.. [Cheng2011] Cheng J. et al., "Theoretical Analysis and Pratical Insights on
   EAP Estimation via Unified HARDI Framework", MICCAI workshop on
   Computational Diffusion MRI, 2011.

.. include:: ../links_names.inc

"""
