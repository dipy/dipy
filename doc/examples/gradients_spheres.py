"""
=====================
Gradients and Spheres
=====================

This example shows how you can create GradientTables and Sphere objects using 
Dipy.

Usually, as we saw in :ref:`example_quick_start` you load your b-values and 
b-vectors from the disk and then you can create your own GradientTable. But,
this time lets say that you are an MR physicist and you want to desing a new
gradient scheme or you are a scientist who wants to simulate many different
gradient schemes. 

Okay, now let's assume that you are interested in creating a multi-shell 
acquisition with 2-shells, one at b=1000 and one at b=2500. For both shells
let's say we want a specific number of gradients (64) and we want to have the 
points on the sphere evenly distributed. 

This is possible using the ``disperse_charges`` which is an implementation of
electrostatic repulsion.
"""

import numpy as np
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere

"""
We can first create some random points on a ``HemiSphere`` using spherical polar
coordinates.
"""

n_pts = 64
theta = np.pi * np.random.rand(n_pts)
phi = 2 * np.pi * np.random.rand(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)

"""
Next, we call `disperse_charges` which iteratively will move the points so that
the electrostatic potential energy is minimized.
"""

hsph_updated, potential = disperse_charges(hsph_initial, 5000)

"""
In ``hsph_updated` we have the updated HemiSphere with the points nicely 
distributed on the hemisphere. Let's visualize them.
"""

from dipy.viz import fvtk
ren = fvtk.ren()
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.point(hsph_initial.vertices, fvtk.colors.red, point_radius=0.05))
fvtk.add(ren, fvtk.point(hsph_updated.vertices, fvtk.colors.green, point_radius=0.05))

print('Saving illustration as initial_vs_updated.png')
fvtk.record(ren, out_path='initial_vs_updated.png', size=(300, 300))

"""
.. figure:: initial_vs_updated.png
   :align: center

   **Example of electrostatic repulsion of red points which become green points**.

We can also create a sphere from the hemisphere and show it in the following way.
"""

sph = Sphere(xyz = np.vstack((hsph_updated.vertices, -hsph_updated.vertices)))

fvtk.rm_all(ren)
fvtk.add(ren, fvtk.point(sph.vertices, fvtk.colors.green, point_radius=0.05))

print('Saving illustration as full_sphere.png')
fvtk.record(ren, out_path='full_sphere.png', size=(300, 300))

"""
.. figure:: full_sphere.png
   :align: center

   **Full sphere**

Okay, time to create the Gradients. For this reason we will need to use the
function ``gradient_table`` and fill it with the ``hsph_updated`` vectors that 
we created above.
"""

from dipy.core.gradients import gradient_table

vertices = hsph_updated.vertices
values = np.ones(vertices.shape[0])

"""
So, we need to stacks of ``vertices`` one for every shell and we need two sets
of b-values one at 1000 and one at 2500 as we discussed previously.
"""

bvecs = np.vstack((vertices, vertices))
bvals = np.hstack((1000 * values, 2500 * values))

"""
We can also add some b0s. Let's add one in the beginning and one at the end.
"""

bvecs = np.insert(bvecs, (0, bvecs.shape[0]), np.array([0, 0, 0]), axis=0)
bvals = np.insert(bvals, (0, bvals.shape[0]), 0)

print(bvals)

"""

::

    [    0.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.
      1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.
      1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.
      1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.
      1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.
      1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.  1000.
      1000.  1000.  1000.  1000.  1000.  2500.  2500.  2500.  2500.  2500.
      2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.
      2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.
      2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.
      2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.
      2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.
      2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.  2500.     0.]

"""

print(bvecs)

"""

::

    [[ 0.          0.          0.        ]
     [-0.80451777 -0.16877559  0.56944355]
     [ 0.32822557 -0.94355999  0.04430036]
     [-0.23584135 -0.96241331  0.13468285]
     [-0.39207424 -0.73505312  0.55314981]
     [-0.32539386 -0.16751384  0.93062235]
     [-0.82043195 -0.39411534  0.41420347]
     [ 0.65741493  0.74947875  0.07802061]
     [ 0.88853765  0.45303621  0.07251925]
     [ 0.39638642 -0.15185138  0.90543855]
     [ 0.1169375  -0.24160695  0.9633025 ]
     [ 0.72281481  0.43741645  0.53498187]
     [-0.60824281 -0.63515708  0.47604219]
     [-0.66815897  0.09215793  0.73828891]
     [-0.94742607 -0.02369056  0.31909653]
     [-0.91353908  0.28384923  0.2913348 ]
     [ 0.0557204  -0.98125251  0.18449596]
     [-0.06843661 -0.57080388  0.81822941]
     [ 0.94491235 -0.32498512  0.03905543]
     [ 0.5431459   0.70127546  0.46174155]
     [-0.09772376  0.86273581  0.49612195]
     [-0.31438857  0.48933796  0.81345448]
     [-0.40855143  0.16658388  0.89740489]
     [ 0.64762701  0.25514034  0.71797121]
     [ 0.2204986  -0.50537377  0.83425279]
     [ 0.3695973   0.91053664  0.18525889]
     [-0.11700816  0.34295503  0.93203591]
     [-0.20872248  0.69524954  0.68779576]
     [ 0.6494387  -0.75529251  0.08810562]
     [ 0.12528696  0.44733607  0.88554707]
     [ 0.34937552  0.14232056  0.92611101]
     [ 0.4153057  -0.63535592  0.65103305]
     [-0.40190996  0.89204544  0.20669618]
     [-0.42533691  0.78382644  0.45244295]
     [-0.1929142  -0.3846751   0.90266781]
     [-0.9585949  -0.25187212  0.13287683]
     [-0.78247393 -0.59083739  0.19658517]
     [ 0.9983299  -0.05083648  0.0274421 ]
     [ 0.0708047   0.95927477  0.27345679]
     [-0.8035861   0.23303438  0.54767176]
     [ 0.04780422 -0.75292905  0.65636316]
     [ 0.62613015 -0.61577961  0.47830588]
     [-0.14668674  0.01999407  0.98898091]
     [ 0.75176554 -0.20762793  0.62589074]
     [-0.67849786  0.57839153  0.45288397]
     [ 0.2258399  -0.84652754  0.48206583]
     [ 0.82574745 -0.54011117  0.16254563]
     [ 0.95421007  0.1966386   0.22542494]
     [ 0.80278668 -0.40709285  0.4356707 ]
     [ 0.62528869 -0.01720989  0.78020374]
     [-0.52544851 -0.82790532  0.19615466]
     [ 0.25913357  0.63696923  0.7260303 ]
     [-0.57487382  0.42529904  0.69902848]
     [ 0.4639848  -0.82985657  0.30992932]
     [ 0.76790335  0.56596022  0.30002245]
     [-0.17020785 -0.82471935  0.53932113]
     [ 0.85157207  0.12964048  0.50795508]
     [-0.79313851  0.58411973  0.17243971]
     [-0.58583154 -0.16596913  0.79325636]
     [ 0.92830919 -0.12961488  0.34848535]
     [ 0.10175269  0.08197111  0.99142681]
     [ 0.50577702 -0.37862345  0.77513476]
     [ 0.42845026  0.40155296  0.80943535]
     [ 0.26939707  0.81103868  0.51927014]
     [-0.48938584 -0.43780086  0.75420946]
     [-0.80451777 -0.16877559  0.56944355]
     [ 0.32822557 -0.94355999  0.04430036]
     [-0.23584135 -0.96241331  0.13468285]
     [-0.39207424 -0.73505312  0.55314981]
     [-0.32539386 -0.16751384  0.93062235]
     [-0.82043195 -0.39411534  0.41420347]
     [ 0.65741493  0.74947875  0.07802061]
     [ 0.88853765  0.45303621  0.07251925]
     [ 0.39638642 -0.15185138  0.90543855]
     [ 0.1169375  -0.24160695  0.9633025 ]
     [ 0.72281481  0.43741645  0.53498187]
     [-0.60824281 -0.63515708  0.47604219]
     [-0.66815897  0.09215793  0.73828891]
     [-0.94742607 -0.02369056  0.31909653]
     [-0.91353908  0.28384923  0.2913348 ]
     [ 0.0557204  -0.98125251  0.18449596]
     [-0.06843661 -0.57080388  0.81822941]
     [ 0.94491235 -0.32498512  0.03905543]
     [ 0.5431459   0.70127546  0.46174155]
     [-0.09772376  0.86273581  0.49612195]
     [-0.31438857  0.48933796  0.81345448]
     [-0.40855143  0.16658388  0.89740489]
     [ 0.64762701  0.25514034  0.71797121]
     [ 0.2204986  -0.50537377  0.83425279]
     [ 0.3695973   0.91053664  0.18525889]
     [-0.11700816  0.34295503  0.93203591]
     [-0.20872248  0.69524954  0.68779576]
     [ 0.6494387  -0.75529251  0.08810562]
     [ 0.12528696  0.44733607  0.88554707]
     [ 0.34937552  0.14232056  0.92611101]
     [ 0.4153057  -0.63535592  0.65103305]
     [-0.40190996  0.89204544  0.20669618]
     [-0.42533691  0.78382644  0.45244295]
     [-0.1929142  -0.3846751   0.90266781]
     [-0.9585949  -0.25187212  0.13287683]
     [-0.78247393 -0.59083739  0.19658517]
     [ 0.9983299  -0.05083648  0.0274421 ]
     [ 0.0708047   0.95927477  0.27345679]
     [-0.8035861   0.23303438  0.54767176]
     [ 0.04780422 -0.75292905  0.65636316]
     [ 0.62613015 -0.61577961  0.47830588]
     [-0.14668674  0.01999407  0.98898091]
     [ 0.75176554 -0.20762793  0.62589074]
     [-0.67849786  0.57839153  0.45288397]
     [ 0.2258399  -0.84652754  0.48206583]
     [ 0.82574745 -0.54011117  0.16254563]
     [ 0.95421007  0.1966386   0.22542494]
     [ 0.80278668 -0.40709285  0.4356707 ]
     [ 0.62528869 -0.01720989  0.78020374]
     [-0.52544851 -0.82790532  0.19615466]
     [ 0.25913357  0.63696923  0.7260303 ]
     [-0.57487382  0.42529904  0.69902848]
     [ 0.4639848  -0.82985657  0.30992932]
     [ 0.76790335  0.56596022  0.30002245]
     [-0.17020785 -0.82471935  0.53932113]
     [ 0.85157207  0.12964048  0.50795508]
     [-0.79313851  0.58411973  0.17243971]
     [-0.58583154 -0.16596913  0.79325636]
     [ 0.92830919 -0.12961488  0.34848535]
     [ 0.10175269  0.08197111  0.99142681]
     [ 0.50577702 -0.37862345  0.77513476]
     [ 0.42845026  0.40155296  0.80943535]
     [ 0.26939707  0.81103868  0.51927014]
     [-0.48938584 -0.43780086  0.75420946]
     [ 0.          0.          0.        ]]

Both b-values and b-vectors look correct. Let's know create the 
``GradientTable``.
"""

gtab = gradient_table(bvals, bvecs)

fvtk.rm_all(ren)

"""
We can also visualize the gradients. Let's color with blue the first shell and
with cyan the second shell.
"""

colors_b1000 = fvtk.colors.blue * np.ones(vertices.shape)
colors_b2500 = fvtk.colors.cyan * np.ones(vertices.shape)
colors = np.vstack((colors_b1000, colors_b2500))
colors = np.insert(colors, (0, colors.shape[0]), np.array([0, 0, 0]), axis=0)
colors = np.ascontiguousarray(colors)

fvtk.add(ren, fvtk.point(gtab.gradients, colors, point_radius=100))

print('Saving illustration as gradients.png')
fvtk.record(ren, out_path='gradients.png', size=(300, 300))

"""
.. figure:: gradients.png
   :align: center

   **Diffusion Gradients**
"""

