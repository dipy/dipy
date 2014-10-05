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

model_predict = FF.predict()

"""

We will focus on the error in prediction of the diffusion-weighted data, and
calculate the root of the mean squared error.

"""

model_error = model_predict - FF.data
model_rmse = np.sqrt(np.mean(model_error[:, 10:] ** 2, -1))


"""

As a baseline against which we can compare, we calculate another error term
based on the naive prediction that all the tracks are necessary and they each
contribute equally to the signal. In each voxel, the predictions are divided by
the number of tracks in that voxel, so that things would not get out of hand

"""

sum_signals = np.asarray(FF.life_matrix.sum(-1)).squeeze()
tracks_per_voxel = np.asarray(FF.life_matrix.astype(bool).sum(axis=-1)).squeeze()

tracks_prediction = sum_signals/tracks_per_voxel



tracks_prediction = np.reshape(tracks_prediction,
                              (FF.vox_coords.shape[0],np.sum(~gtab.b0s_mask)))


"""

Since the fitting is done in the demeaned S/S0 domain, we need
to add back the mean and then multiply by S0 in every voxel:

"""

tracks_prediction = ( (tracks_prediction + FF.mean_signal[:, None]) *
                      FF.b0_signal[:, None])

tracks_error = tracks_prediction - FF.data[:, ~gtab.b0s_mask]
tracks_rmse = np.sqrt(np.mean(tracks_error ** 2, -1))

"""

First, we can compare the overall distribution of errors between these two
alternative models of the ROI. We show the distribution of differences in error
(improvement through model fitting, relative to the baseline model). Here,
positive values denote an improvement in error with model fit, relative to
without the model fit

"""

import matplotlib.pyplot as plt
import matplotlib

fig, ax = plt.subplots(1)
ax.hist(tracks_rmse - model_rmse, bins=100, histtype='step')
ax.text(0.2, 0.9,'Median RMSE, tracks: %.2f' % np.median(tracks_rmse),
     horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes)
ax.text(0.2, 0.8,'Median RMSE, LiFE: %.2f' % np.median(model_rmse),
     horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes)

fig.savefig('error_histograms.png')


"""

.. figure:: error_histograms.png
   :align: center

   **Improvement in error with fitting of the LiFE model**.

"""


"""

Second, we can show the spatial distribution of the two error terms,
and of the improvement with the model fit:

"""

vol_model = np.ones(data.shape[:3]) * np.nan
vol_model[FF.vox_coords[:, 0], FF.vox_coords[:, 1], FF.vox_coords[:, 2]] =\
                                                                  model_rmse
vol_tracks = np.ones(data.shape[:3]) * np.nan
vol_tracks[FF.vox_coords[:, 0], FF.vox_coords[:, 1], FF.vox_coords[:, 2]] =\
                                                                  tracks_rmse

vol_improve = np.ones(data.shape[:3]) * np.nan
vol_improve[FF.vox_coords[:, 0], FF.vox_coords[:, 1], FF.vox_coords[:, 2]] =\
                                                        tracks_rmse - model_rmse


sl_idx = 49
from mpl_toolkits.axes_grid1 import AxesGrid
fig = plt.figure()
fig.subplots_adjust(left=0.05, right=0.95)
ax = AxesGrid(fig, 111,
              nrows_ncols = (1, 3),
              label_mode = "1",
              share_all = True,
              cbar_location="top",
              cbar_mode="each",
              cbar_size="10%",
              cbar_pad="5%")

ax[0].matshow(np.rot90(t1_data[sl_idx, :, :]), cmap=matplotlib.cm.bone)
im = ax[0].matshow(np.rot90(vol_model[sl_idx, :, :]), cmap=matplotlib.cm.hot)
ax.cbar_axes[0].colorbar(im)
ax[1].matshow(np.rot90(t1_data[sl_idx, :, :]), cmap=matplotlib.cm.bone)
im = ax[1].matshow(np.rot90(vol_tracks[sl_idx, :, :]), cmap=matplotlib.cm.hot)
ax.cbar_axes[1].colorbar(im)
ax[2].matshow(np.rot90(t1_data[sl_idx, :, :]), cmap=matplotlib.cm.bone)
im = ax[2].matshow(np.rot90(vol_improve[sl_idx, :, :]), cmap=matplotlib.cm.hot)
ax.cbar_axes[2].colorbar(im)


for lax in ax:
    lax.set_xticks([])
    lax.set_yticks([])

fig.savefig("spatial_errors.png")

"""

.. figure:: spatial_errors.png
   :align: center

   **The spatial distribution of error and improvement **

"""



"""

.. [Pestilli2014] Pestilli, F., Yeatman, J, Rokem, A. Kay, K. and Wandell
                  B.A. (2014). Validation and statistical inference in living
                  connectomes. Nature Methods 11:
                  1058-1063. doi:10.1038/nmeth.3098

.. include:: ../links_names.inc


"""
