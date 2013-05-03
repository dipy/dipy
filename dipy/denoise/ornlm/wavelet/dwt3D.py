import numpy as np
from afb3D import afb3D
def dwt3D(x, J, af):
    w=[None]*(J+1)
    for k in xrange(J):
        x, w[k] = afb3D(x, af, af, af);
    w[J] = x;
    return w