"""
=================================================
Linear fascicle evaluation (LiFE)
=================================================

Evaluating the results of tractography algorithms is one of the biggest
challenges for diffusion MRI. One proposal for evaluation of tractography
results is to use a forward model that predicts the signal from each of a set of
tracks, and then fit a linear model to these simultaneous predictions
[Pestilli2014]_.

We will use tracks generated using probabilistic tracking on CSA peaks. For
brevity, we will include in this example only tracks going through the corpus
callosum connecting left to right superior frontal cortex. The process of
tracking and finding these tracks is fully demonstrated in the
`streamline_tools.py` example. If this example has been run, we can read the
streamlines from file. Otherwise, we'll run that example first, by importing
it. This provides us with all of the variables that were created in that
example:
"""
import numpy as np
import os.path as op
import nibabel as nib

if not op.exists('lr-superiorfrontal.trk'):
    from streamline_tools import *
else:
    # We'll need to know where the corpus callosum is from these variables:
    from dipy.data import (read_stanford_labels, fetch_stanford_t1,
                           read_stanford_t1)
    hardi_img, gtab, labels_img = read_stanford_labels()
    labels = labels_img.get_data()
    cc_slice = labels == 2
    t1 = read_stanford_t1()
    t1_data = t1.get_data()
    data = hardi_img.get_data()

# Read the candidates from file in voxel space:
candidate_sl = [s[0] for s in nib.trackvis.read('lr-superiorfrontal.trk',
                                                  points_space='voxel')[0]]

"""
The tracks that are entered into the model are termed 'candidate tracks' (or a
'candidate connectome'):
"""


"""
Let's visualize the initial candidate group of tracks in 3D, relative to the
anatomical structure of this brain:
"""

from dipy.viz.colormap import line_colors
from dipy.viz import fvtk
candidate_streamlines_actor = fvtk.streamtube(candidate_sl,
                                       line_colors(candidate_sl))
cc_ROI_actor = fvtk.contour(cc_slice, levels=[1], colors=[(1., 1., 0.)],
                            opacities=[1.])

vol_actor = fvtk.slicer(t1_data, voxsz=(1.0, 1.0, 1.0), plane_i=[40],
                        plane_j=None, plane_k=[35], outline=False)

# Add display objects to canvas
ren = fvtk.ren()
fvtk.add(ren, candidate_streamlines_actor)
fvtk.add(ren, cc_ROI_actor)
fvtk.add(ren, vol_actor)

fvtk.record(ren, n_frames=1, out_path='life_candidates.png',
            size=(800, 800))

"""
.. figure:: life_candidates.png
   :align: center

   **Candidate connectome before life optimization**

"""


"""

Next, we initialize a LiFE model. We import the `dipy.tracking.life` module,
which contains the classes and functions that implement the model:

"""
import dipy.tracking.life as life
FM = life.FiberModel(gtab)

"""

Since we read the tracks from a file, already in the voxel space, we do not
need to transform them into this space. Otherwise, if the track coordinates
were in the world space (relative to the scanner iso-center, or relative to the
mid-point of the AC-PC-connecting line), we would use this::

   inv_affine = np.linalg.inv(hardi_img.get_affine())

the inverse transformation from world space to the voxel space as the affine for
the following model fit.

The next step is to fit the model, producing a `FiberFit` class instance, that
stores the data, as well as the results of the fitting procedure.

The LiFE model posits that the signal in the diffusion MRI volume can be
explained by the tracks, by the equation

.. math::

    y = X\beta


Where $y$ is the diffusion MRI signal, $beta$ are a set of weights on the
tracks and $X$ is a design matrix. This matrix has the dimensions $m$ by $n$,
where $m=#_{voxels} #_{directions}$, and $#_{voxels}$ is the set of voxels in
the ROI that contains the tracks considered in this model. The $i^{th}$ column
of the matrix contains the expected contributions of the $i^{th}$ track
(arbitrarly ordered) to each of the voxels. $X$ is a sparse matrix, becasue
each track traverses only a small percentage of the voxels. The expected
contributions of the track are calculated using a forward model, where each
node of the track is modeled as a cylindrical fiber compartment with Gaussian
diffusion, using the diffusion tensor model. See [Pestilli2014]_ for more
detail on the model, and variations of this model.

"""

FF = FM.fit(data, candidate_sl, affine=np.eye(4))

"""

The `FiberFit` class instance holds various properties of the model fit. For
example, it has the weights $\beta$, that are assigned to each track. In most
cases, a tractography through some region will include redundant tracks, and
these tracks will have $\beta_i$ that are 0. We use $\beta$ to filter out these
redundant tracks, and generate an optimized group of streamlines:

"""

optimized_sl = list(np.array(candidate_sl)[np.where(FF.beta>0)[0]])
ren = fvtk.ren()
fvtk.add(ren, fvtk.streamtube(optimized_sl, line_colors(optimized_sl)))
fvtk.add(ren, cc_ROI_actor)
fvtk.add(ren, vol_actor)
fvtk.record(ren, n_frames=1, out_path='life_optimized.png',
            size=(800, 800))

"""

.. figure:: life_optimized.png
   :align: center

   **Tracks selected via life optimization**

"""

"""

How well does the model do in explaining the diffusion data? The `FiberFit`
class instance has a `predict` method, which can be used to invert the model
and predict back either the data that was used to fit the model, or other
unseen data (e.g. in cross-validation, see :kfold_xval:).



Without arguments, the `.predict()` method will predict the diffusion signal
for the same gradient table that was used in the fit data, but `gtab` and `S0`
key-word arguments can be used to predict for other acquisition schemes and
other non diffusion-weighted signals.

"""

predict = FF.predict()

"""

We will focus on the error in prediction of the diffusion-weighted data, and
calculate the root of the mean squared error. We can demonstrate the reduction
of the error that is afforded by the fitting of the model by calculating two
error terms. The

"""

model_error = predict - FF.data
model_rmse = np.sqrt(np.mean(model_error[:, 10:] ** 2, -1))

tracks_error = np.reshape(
               life.spdot(FF.life_matrix, np.ones(FF.life_matrix.shape[-1])),
               (FF.vox_coords.shape[0], np.sum(~gtab.b0s_mask)))

tracks_rmse = np.sqrt(np.mean(tracks_error[:, 10:] ** 2, -1))

"""

The second error term is the error in matching the data based on uniform
weights on the entire set of candidate tracks. This can be derived by
multiplying out the

"""

import matplotlib.pyplot as plt

vol = np.ones(data.shape[:3]) * np.nan

fig, ax = plt.subplots(3)
ax.imshow(t1_data[49, :, :], cmap=matplotlib.cm.bone)
imshow(vol[49, :, :], cmap=matplotlib.cm.hot)
import matplotlib.pyplot as plt



"""



"""



"""

.. [Pestilli2014] Pestilli, F., Yeatman, J, Rokem, A. Kay, K. and Wandell
                  B.A. (2014). Validation and statistical inference in living
                  connectomes. Nature Methods 11:
                  1058-1063. doi:10.1038/nmeth.3098

.. include:: ../links_names.inc


"""
