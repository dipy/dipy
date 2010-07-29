import numpy as np

import dipy.core.meshes as meshes

from dipy.core.triangle_subdivide import create_unit_sphere, remove_half_sphere
v, e, t = create_unit_sphere(5)
vH, eH, tH = remove_half_sphere(v, e, t)




