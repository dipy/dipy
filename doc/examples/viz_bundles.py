"""
========================================
Visualize bundles and metrics on bundles
========================================

First, let's download some available datasets. Here we are using a dataset
which provides metrics and bundles.
"""

import numpy as np

from dipy.data import fetch_bundles_2_subjects, read_bundles_2_subjects
from dipy.tracking.streamline import length, values_from_volume
from dipy.viz import actor
from dipy.viz.streamline import show_bundles

fetch_bundles_2_subjects()
dix = read_bundles_2_subjects(
    subj_id="subj_1", metrics=["fa"], bundles=["cg.left", "cst.right"]
)

###############################################################################
# Store fractional anisotropy.

fa = dix["fa"]

###############################################################################
# Store grid to world transformation matrix.

affine = dix["affine"]

###############################################################################
# Store the cingulum bundle. A bundle is a list of streamlines.

bundle = dix["cg.left"]

###############################################################################
# It happened that this bundle is in world coordinates. ``show_bundles`` draws
# in world coordinates, so the bundle is used as is. The ``affine`` is only
# needed to sample the ``fa`` image along the streamlines.

###############################################################################
# Show every streamline with an orientation color
# ===============================================
#
# This is the default option of ``show_bundles``: when ``colors`` is not given,
# each streamline gets one RGB color computed from the direction between its
# end points. ``show_bundles`` takes a list of bundles, so we pass ``[bundle]``.
# With ``interactive=False`` the scene is only saved into the png ``save_as``;
# use ``interactive=True`` to open a window instead.

show_bundles([bundle], interactive=False, save_as="bundle1.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# One orientation color for every streamline.
#
#
# Show every point with a value from a volume with default colormap
# =================================================================
#
# ``values_from_volume`` samples ``fa`` at every point of the bundle, using
# ``affine`` to go from world to voxel coordinates. The values are concatenated
# in the order of the bundle points, so there is one value per point.

point_fa = np.concatenate(values_from_volume(fa, bundle, affine))

###############################################################################
# The scalar bar built with ``actor.scalar_bar`` maps the values to colors and
# shows that mapping in the figure. Its ``map_values`` method gives one RGB
# color per value. The default colormap goes from purple (FA 0) to red (FA 1)
# through blue, green and yellow: its hue goes from 0.8 down to 0. ``colors``
# holds one entry per bundle, here a single array with one color per point.

hue = (0.8, 0.0)  # purple to red
saturation = (1.0, 1.0)  # fully saturated
value = (0.9, 0.9)  # slightly dimmed

fa_bar = actor.scalar_bar(
    clim=(0.0, 1.0),
    hsv_range=(hue, saturation, value),
    title="FA",
    tick_format=".2f",
    color=(0.0, 0.0, 0.0),
    position=(200, 60),
)

show_bundles(
    [bundle],
    colors=[fa_bar.map_values(point_fa)],
    scalar_bar=fa_bar,
    interactive=False,
    save_as="bundle2.png",
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Every point with a color from FA.
#
#
# Show every point with a value from a volume with your colormap
# ==============================================================
#
# The colormap is given as hue, saturation and value ramps, from the lowest to
# the highest FA.

hue = (0.0, 0.0)  # red only
saturation = (0.0, 1.0)  # white to red
value = (0.9, 0.9)  # slightly dimmed

red_bar = actor.scalar_bar(
    clim=(0.0, 1.0),
    hsv_range=(hue, saturation, value),
    title="FA",
    tick_format=".2f",
    color=(0.0, 0.0, 0.0),
    position=(200, 60),
)

show_bundles(
    [bundle],
    colors=[red_bar.map_values(point_fa)],
    scalar_bar=red_bar,
    interactive=False,
    save_as="bundle3.png",
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Every point with a color from FA using a non default colormap.
#
#
# Show every bundle with a specific color
# ========================================
#
# You can have a bundle with a specific color. In this example, we are choosing
# orange.

show_bundles(
    [bundle], colors=[(1.0, 0.5, 0.0)], interactive=False, save_as="bundle4.png"
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Entire bundle with a specific color.
#
#
# Show every streamline of a bundle with a different color
# ========================================================
#
# Let's make a colormap where every streamline of the bundle is colored by
# its length. This time there is one color per streamline.

lengths = length(bundle)

hue = (0.5, 0.5)  # cyan only
saturation = (0.0, 1.0)  # white to cyan
value = (0.9, 0.9)  # slightly dimmed

length_bar = actor.scalar_bar(
    clim=(lengths.min(), lengths.max()),
    hsv_range=(hue, saturation, value),
    title="Length (mm)",
    tick_format=".0f",
    color=(0.0, 0.0, 0.0),
    position=(200, 60),
)

show_bundles(
    [bundle],
    colors=[length_bar.map_values(lengths)],
    scalar_bar=length_bar,
    interactive=False,
    save_as="bundle5.png",
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Color every streamline by the length of the streamline
#
#
# Show every point of every streamline with a different color
# ============================================================
#
# In this case in which we want to have a color per point and per streamline,
# we can create one random RGB color per point, as a single array ordered like
# the points of the bundle.

rng = np.random.default_rng()

colors = rng.random((bundle.total_nb_rows, 3))

show_bundles([bundle], colors=[colors], interactive=False, save_as="bundle6.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Random colors per point per streamline.
#
#
# In summary, we showed that there are many useful ways for visualizing maps
# on bundles.
