import warnings

from dipy.direction.peaks import *

w_s = "The module 'dipy.reconst.peaks' is deprecated."
w_s += " Please use the module 'dipy.direction.peaks' instead"
warnings.warn(w_s, DeprecationWarning)
