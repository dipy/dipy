import numpy as np
from dipy.core.reconstruction_performance import peak_finding
from dipy.core.reconstruction_performance import pf_bago

from dipy.core.triangle_subdivide import create_unit_sphere, remove_half_sphere

v, e, t = create_unit_sphere(7)
vH, eH, tH = remove_half_sphere(v, e, t)

odf = np.random.random(len(vH))

pB, iB = pf_bago(odf, eH)

bO = np.r_[odf, odf]
faces = e[t,0]
faces = faces/2 + (faces % 2)*len(odf)

p, i = peak_finding(bO, faces)



