"""
================================
BUAN bundle profiles (lite)
================================

This example shows how to compute **weighted mean bundle profiles**
using the BUAN "lite" functionality.

Compared to full BUAN profiles (streamline-by-streamline), the lite version
summarizes each bundle into a mean profile, where each streamline contributes
according to a weight based on its distance to a model bundle centroid.

The output is useful for users who prefer working with **mean profiles**
while keeping BUAN-style along-tract segmentation.

"""

import numpy as np

from dipy.data import fetch_bundles_2_subjects, read_bundles_2_subjects
from dipy.tracking.streamline import length, transform_streamlines
from dipy.viz import actor, window

fetch_bundles_2_subjects()
dix = read_bundles_2_subjects(
    subj_id="subj_1", metrics=["fa"], bundles=["cst.right"]
)

###############################################################################
# Store fractional anisotropy.

fa = dix["fa"]

###############################################################################
# Store grid to world transformation matrix.

affine = dix["affine"]

###############################################################################
# Store the corticospinal tract. A bundle is a list of streamlines.

bundle = dix["CST.left"]

