from dipy.io.utils import decfa, decfa_to_float
from nibabel import Nifti1Image
import numpy as np


def test_decfa():
    data_orig = np.zeros((4, 4, 4, 3))
    data_orig[0, 0, 0] = np.array([1, 0, 0])
    img_orig = Nifti1Image(data_orig, np.eye(4))
    img_new = decfa(img_orig)
    data_new = img_new.get_data()
    assert data_new[0, 0, 0] == np.array((1, 0, 0),
                                         dtype=np.dtype([('R', 'uint8'),
                                                         ('G', 'uint8'),
                                                         ('B', 'uint8')]))
    assert data_new.dtype == np.dtype([('R', 'uint8'),
                                       ('G', 'uint8'),
                                       ('B', 'uint8')])

    round_trip = decfa_to_float(img_new)
    data_rt = round_trip.get_fdata()
    assert np.all(data_rt == data_orig)

    data_orig = np.zeros((4, 4, 4, 3))
    data_orig[0, 0, 0] = np.array([0.1, 0, 0])
    img_orig = Nifti1Image(data_orig, np.eye(4))
    img_new = decfa(img_orig, scale=True)
    data_new = img_new.get_data()
    assert data_new[0, 0, 0] == np.array((25, 0, 0),
                                         dtype=np.dtype([('R', 'uint8'),
                                                         ('G', 'uint8'),
                                                         ('B', 'uint8')]))
    assert data_new.dtype == np.dtype([('R', 'uint8'),
                                       ('G', 'uint8'),
                                       ('B', 'uint8')])

    round_trip = decfa_to_float(img_new)
    data_rt = round_trip.get_data()
    assert data_rt.shape == (4, 4, 4, 3)
    assert np.all(data_rt[0, 0, 0] == np.array([25, 0, 0]))
