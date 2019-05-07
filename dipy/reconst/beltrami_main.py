"""
Implements the main gradient descent function to estimate
the Free Water parameter from single-shell diffusion data.
"""

from __future__ import division
import numpy as np
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import design_matrix
import beltrami as blt  # importing the functions from beltrami.py
