"""
================================================================
Continuous and analytical diffusion signal modelling with MAPMRI
================================================================

We show how to model the diffusion signal as a linear combination
of continuous functions from the MAPMRI basis [Ozarslan2013]_.
We also compute the analytical Orientation Distribution Function (ODF),
the the Return To the Origin Probability (RTOP), the Return To the Axis
Probability (RTAP), and the Return To the Plane Probability (RTPP).

First import the necessary modules:
"""

from dipy.reconst.mapmri import MapmriModel
from dipy.viz import fvtk
from dipy.data import fetch_cenir_multib, read_cenir_multib, get_sphere
from dipy.core.gradients import gradient_table
import matplotlib.pyplot as plt

"""
Download and read the data for this tutorial.

MAPMRI requires multi-shell data, to properly fit the radial part of the basis.
The total size of the downloaded data is 1760 MBytes, however you only need to
fetch it once. Parameter ``with_raw`` of function ``fetch_cenir_multib`` is set
to ``False`` to only download eddy-current/motion corrected data:.
"""

fetch_cenir_multib(with_raw=False)

"""
For this example we select only the shell with b-values equal to the one of the
Human Connectome Project (HCP).
"""
bvals = [1000, 2000, 3000]
img, gtab = read_cenir_multib(bvals)
data = img.get_data()
data_small = data[40:65, 50:51, 35:60]

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
map_model = MapmriModel(gtab, radial_order=radial_order,
                        lambd=2e-1, eap_cons=False)

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
print('odf.shape (%d, %d, %d, %d)' % odf.shape)

"""
Display the ODFs
"""

r = fvtk.ren()
sfu = fvtk.sphere_funcs(odf, sphere, colormap='jet')
sfu.RotateX(-90)
fvtk.add(r, sfu)
fvtk.record(r, n_frames=1, out_path='odfs.png', size=(600, 600))

"""
.. figure:: odfs.png
   :align: center

   **Orientation distribution functions**.

With MAPMRI it is also possible to extract the Return To the Origin Probability
(RTOP), the Return To the Axis Probability (RTAP), and the Return To the Plane
Probability (RTPP). These ensemble average propagator (EAP) features directly
reflects microstructural properties of the underlying tissues [Ozarslan2013]_. 
"""

rtop = mapfit.rtop()
rtap = mapfit.rtap()
rtpp = mapfit.rtpp()

"""
Show the maps and save them in MAPMRI_maps.png.
"""

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(2, 2, 1, title=r'$\sqrt[3]{RTOP}$')
ax1.set_axis_off()
ind = ax1.imshow((rtop[:, 0, :]**(1.0 / 3)).T,
                 interpolation='nearest', origin='lower', cmap=plt.cm.gray)
plt.colorbar(ind, shrink = 0.8)
ax2 = fig.add_subplot(2, 2, 2, title=r'$\sqrt{RTAP}$')
ax2.set_axis_off()
ind = ax2.imshow((rtap[:, 0, :]**0.5).T,
                 interpolation='nearest', origin='lower', cmap=plt.cm.gray)
plt.colorbar(ind, shrink = 0.8)
ax3 = fig.add_subplot(2, 2, 3, title=r'$RTPP$')
ax3.set_axis_off()
ind = ax3.imshow(rtpp[:, 0, :].T, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray)
plt.colorbar(ind, shrink = 0.8)
plt.savefig('MAPMRI_maps.png')

"""
.. figure:: MAPMRI_maps.png
   :align: center

   **RTOP, RTAP, and RTPP calculated using MAPMRI**.

.. [Ozarslan2013] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
               diffusion imaging method for mapping tissue microstructure",
               NeuroImage, 2013.

.. include:: ../links_names.inc

"""
