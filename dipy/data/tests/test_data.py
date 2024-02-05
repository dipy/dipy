import numpy.testing as npt
from dipy.data import SPHERE_FILES
import numpy as np


def test_sphere_dtypes():
    for sphere_name, sphere_path in SPHERE_FILES.items():
        sphere_data = np.load(sphere_path)
        npt.assert_equal(sphere_data['vertices'].dtype, np.dtype('<f8'))
