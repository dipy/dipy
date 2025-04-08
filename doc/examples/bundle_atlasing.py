"""
===============
Bundle Atlasing
===============

Bundle Atlasing is the DIPY framework for creating streamline atlases from
previously segmented bundles of multiple subjects :footcite:p:`RomeroBascones2022`.

How it works
============

The atlasing procedure is implemented in the ``compute_atlas_bundle()`` function,
which iteratively combines the input bundles in pairs using a tree structure.
"""

###############################################################################
# The input bundles are expected to be stored in a BIDS-like structure where the
# segmented bundles of each subject are stored in a different subfolder. Note that
# the naming of the bundles must be the same for all subjects::
#
#     in_dir/
#     ├── sub_1/
#     │   ├── AF_L.trk
#     │   ├── CST_R.trk
#     │   └── CC_ForcepsMajor.trk
#     ├── sub_2/
#     │   ├── AF_L.trk
#     │   ├── CST_R.trk
#     │   └── CC_ForcepsMajor.trk
#     ...
#     └── sub_N/
#         ├── AF_L.trk
#         ├── CST_R.trk
#         └── CC_ForcepsMajor.trk

"""
If there are more folder levels between subject folders and the bundles, the
``mid_path`` parameter can be used to specify the path to the bundles.
"""

###############################################################################
# For instance, in the below example, ``mid_path`` should be ``"bundles"``::
#
#     in_dir/
#     ├── sub_1/
#     │   ├── T1/
#     │   └── bundles/
#     │   │   ├── AF_L.trk
#     │   │   ├── CST_R.trk
#     │   │   ├── CC_ForcepsMajor.trk
#     ...
#     └── sub_N/
#         ├── T1/
#         └── bundles/
#             ├── AF_L.trk
#             ├── CST_R.trk
#             └── CC_ForcepsMajor.trk


"""
Example: Atlas Creation
=======================

We start by importing the necessary functions:
"""

import logging
import tempfile

from dipy.atlasing.bundles import compute_atlas_bundle
from dipy.data import read_five_af_bundles
from dipy.data.fetcher import extract_example_tracts
from dipy.viz.streamline import show_bundles

logging.basicConfig(level=logging.INFO)

###############################################################################
# First, let's create an input directory that will contain example bundles from
# five subjects following the folder structure shown above. The directory
# will be created in the system's default temporary directory and should be
# replaced with the actual path to the bundles to be atlased.

temp_dir = tempfile.mkdtemp()

extract_example_tracts(temp_dir)  # Extract example bundles to temp_dir
bundles = read_five_af_bundles()  # Read AF only bundles

###############################################################################
# The example data contains AF-left, CST-right and forceps major bundles
# from five subjects. Let's visualize the input AF bundles:

colors = [
    (0.91, 0.26, 0.35),
    (0.99, 0.50, 0.38),
    (0.99, 0.88, 0.57),
    (0.69, 0.85, 0.64),
    (0.51, 0.51, 0.63),
]
show_bundles(bundles, interactive=False, colors=colors, save_as="input_bundles.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Input bundles from different subjects.

###############################################################################
# Now we can create atlases for AF_L and CST_R by running:

atlases, _ = compute_atlas_bundle(
    in_dir=temp_dir, bundle_names=["AF_L", "CST_R"], save_temp=True
)

###############################################################################
# The function has two ways of providing the atlas results:
#
# * Returns the atlases as function outputs
# * Saves the atlases as .trk files in an output folder named
#   ``bundle_atlasing_<timestamp>``
#
# By default the function will save the atlases in the current working directory.
# We can choose the output directory by setting the ``out_dir`` parameter.
#
# When ``save_temp=True`` all intermediate results are saved into a folder named
# ``temp`` in the output directory.
#
# We can now visualize the AF_L atlas:

colors = [
    (0.91, 0.26, 0.35),
    (0.99, 0.50, 0.38),
    (0.99, 0.88, 0.57),
    (0.69, 0.85, 0.64),
    (0.51, 0.51, 0.63),
    (0, 0, 1),
]

atlas_AF = atlases[0]  # The first element is the AF_L atlas

show_bundles(
    bundles + [atlas_AF],
    interactive=False,
    colors=colors,
    save_as="atlas_bundle_AF.png",
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Resulting AF atlas bundle.

###############################################################################
# In a similar way, we can visualize the CST_R atlas:

atlas_CST = atlases[1]  # The second element is the CST_R atlas

show_bundles(
    [atlas_CST],
    interactive=False,
    colors=[(0, 0, 1)],
    save_as="atlas_bundle_CST.png",
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Resulting CST atlas bundle.

###############################################################################
# Controlling the atlasing procedure
# ==================================
#
# The smoothness of the atlasing procedure can be controlled by the following
# parameters:
#
#  - ``d_max``: it is the maximum distance between streamlines to be averaged.
#    Smaller values result in less smooth atlases that preserve more sharp features.
#
#  - ``skip_pairs``: if True, the combination of random bundle pairs is skipped thus
#    reducing the amount of averaging and resulting in sharper atlases.
#
# The speed of the atlasing procedure can be controlled by the following parameters:
#
#  - ``n_stream_max``: Maximum number of streamlines per bundle. Bundles with more than
#    ``n_stream_max`` streamlines are downsampled to ``n_stream_max`` by randomly
#    selecting streamlines.
#
#  - ``n_point``: number of points per streamline.
#
#  - ``qbx_thr``: threshold used by Quickbundles clustering before registration.
#
# Setting ``merge_out=True`` will merge the output atlases into a single bundle that
# will be returned as the second output argument.
#
# References
# ----------
#
# .. footbibliography::
#
