from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.testing as npt

from distutils.version import LooseVersion
from dipy.reconst.peaks import PeaksAndMetrics

# Conditional testing machinery for pytables
from dipy.testing import doctest_skip_parser

# Conditional import machinery for pytables
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests, if we don't have pytables
tables, have_tables, _ = optional_package('tables')

# Useful variable for backward compatibility.
if have_tables:
    TABLES_LESS_3_0 = LooseVersion(tables.__version__) < "3.0"

from dipy.data import get_sphere
from dipy.core.sphere import Sphere

# Make sure not to carry across setup module from * import
# __all__ = ['']


def test_io_peaks():

    fname = 'test.pam'

    if TABLES_LESS_3_0:
        func_open_file = tables.openFile
    else:
        func_open_file = tables.open_file

    sphere = get_sphere('repulsion724')
    verbose = True

    pam = PeaksAndMetrics()
    pam.affine = None
    pam.peak_dirs = np.zeros((10, 10, 10, 5, 3))
    pam.peak_values = np.zeros((10, 10, 10, 5))
    pam.peak_indices = np.zeros((10, 10, 10, 5))
    pam.shm_coeff = np.zeros((10, 10, 10, 45))
    pam.sphere = Sphere(xyz=sphere.vertices)
    pam.B = np.zeros((45, sphere.vertices.shape[0]))
    pam.total_weight = 0.5
    pam.ang_thr = 60
    pam.gfa = np.zeros((10, 10, 10))
    pam.qa = np.zeros((10, 10, 10, 5))
    pam.odf = np.zeros((10, 10, 10, sphere.vertices.shape[0]))

    f = func_open_file(fname, 'w')

    if TABLES_LESS_3_0:
        func_create_group = f.createGroup
        func_create_carray = f.createCArray
        func_create_earray = f.createEArray
    else:
        func_create_group = f.create_group
        func_create_carray = f.create_carray
        func_create_earray = f.create_earray

    group = func_create_group(f.root, 'pam')

    def save_array(group, x, name):

        if x is None:
            atom = tables.Atom.from_dtype(np.int)
            ds = func_create_carray(group, name, atom)
        else:
            atom = tables.Atom.from_dtype(x.dtype)
            ds = func_create_carray(group, name, atom, x.shape)
            ds[:] = x

    save_array(group, pam.affine, 'affine')
    save_array(group, pam.peak_dirs, 'peak_dirs')
    save_array(group, pam.peak_values, 'peak_values')
    save_array(group, pam.peak_indices, 'peak_indices')
    save_array(group, pam.shm_coeff, 'shm_coeff')
    save_array(group, pam.sphere.vertices, 'sphere')
    save_array(group, pam.B, 'B')
    save_array(group, np.array([pam.total_weight]), 'total_weight')
    save_array(group, np.array([pam.ang_thr]), 'ang_thr')
    save_array(group, pam.B, 'gfa')
    save_array(group, pam.B, 'qa')
    save_array(group, pam.B, 'odf')

    f.close()

    f2 = func_open_file(fname, 'r')

    print(f2.root.pam.affine)
    f2.close()


    if verbose:
        print('Affine')
        print(pam.affine)
        print('Dirs Shape')
        print(pam.peak_dirs.shape)
        print('SH Shape')
        print(pam.shm_coeff.shape)
        print('ODF')
        print(pam.odf.shape)
        print('Total weight')
        print(pam.total_weight)
        print('Angular threshold')
        print(pam.ang_thr)
        print('Sphere vertices shape')
        print(pam.sphere.vertices.shape)

test_io_peaks()