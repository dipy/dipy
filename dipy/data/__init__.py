"""Read test or example data."""

import json
import pickle

from os.path import join as pjoin, dirname, exists

import gzip
import numpy as np
from scipy.sparse import load_npz
from dipy.core.gradients import GradientTable, gradient_table
from dipy.core.sphere import Sphere, HemiSphere
from dipy.data.fetcher import (get_fnames,
                               fetch_scil_b0,
                               read_scil_b0,
                               fetch_stanford_hardi,
                               read_stanford_hardi,
                               fetch_stanford_tracks,
                               fetch_taiwan_ntu_dsi,
                               read_taiwan_ntu_dsi,
                               fetch_sherbrooke_3shell,
                               read_sherbrooke_3shell,
                               fetch_isbi2013_2shell,
                               read_isbi2013_2shell,
                               read_stanford_labels,
                               fetch_stanford_labels,
                               fetch_syn_data,
                               read_syn_data,
                               fetch_stanford_t1,
                               read_stanford_t1,
                               fetch_stanford_pve_maps,
                               read_stanford_pve_maps,
                               fetch_bundles_2_subjects,
                               read_bundles_2_subjects,
                               fetch_cenir_multib,
                               read_cenir_multib,
                               fetch_mni_template,
                               read_mni_template,
                               fetch_ivim,
                               read_ivim,
                               fetch_tissue_data,
                               read_tissue_data,
                               fetch_cfin_multib,
                               read_cfin_dwi,
                               read_cfin_t1,
                               fetch_target_tractogram_hcp,
                               fetch_bundle_atlas_hcp842,
                               get_bundle_atlas_hcp842,
                               get_target_tractogram_hcp,
                               get_two_hcp842_bundles,
                               fetch_bundle_fa_hcp,
                               fetch_gold_standard_io,
                               fetch_resdnn_weights,
                               fetch_synb0_weights,
                               fetch_synb0_test,
                               fetch_evac_weights,
                               fetch_evac_test,
                               read_qte_lte_pte,
                               read_DiB_70_lte_pte_ste,
                               read_DiB_217_lte_pte_ste,
                               read_five_af_bundles,
                               fetch_hbn,
                               fetch_ptt_minimal_dataset,
                               fetch_bundle_warp_dataset)

from ..utils.arrfuncs import as_native_array
from dipy.io.image import load_nifti
from dipy.tracking.streamline import relist_streamlines


def loads_compat(byte_data):
    return pickle.loads(byte_data, encoding='latin1')


DATA_DIR = pjoin(dirname(__file__), 'files')
SPHERE_FILES = {
    'symmetric362': pjoin(DATA_DIR, 'evenly_distributed_sphere_362.npz'),
    'symmetric642': pjoin(DATA_DIR, 'evenly_distributed_sphere_642.npz'),
    'symmetric724': pjoin(DATA_DIR, 'evenly_distributed_sphere_724.npz'),
    'repulsion724': pjoin(DATA_DIR, 'repulsion724.npz'),
    'repulsion100': pjoin(DATA_DIR, 'repulsion100.npz'),
    'repulsion200': pjoin(DATA_DIR, 'repulsion200.npz')
}


class DataError(Exception):
    pass


def get_sim_voxels(name='fib1'):
    """ provide some simulated voxel data

    Parameters
    ----------
    name : str, which file?
        'fib0', 'fib1' or 'fib2'

    Returns
    -------
    dix : dictionary, where dix['data'] returns a 2d array
        where every row is a simulated voxel with different orientation

    Examples
    --------
    >>> from dipy.data import get_sim_voxels
    >>> sv=get_sim_voxels('fib1')
    >>> sv['data'].shape == (100, 102)
    True
    >>> sv['fibres']
    '1'
    >>> sv['gradients'].shape == (102, 3)
    True
    >>> sv['bvals'].shape == (102,)
    True
    >>> sv['snr']
    '60'
    >>> sv2=get_sim_voxels('fib2')
    >>> sv2['fibres']
    '2'
    >>> sv2['snr']
    '80'

    Notes
    -----
    These sim voxels were provided by M.M. Correia using Rician noise.

    """
    if name == 'fib0':
        fname = pjoin(DATA_DIR, 'fib0.pkl.gz')
    if name == 'fib1':
        fname = pjoin(DATA_DIR, 'fib1.pkl.gz')
    if name == 'fib2':
        fname = pjoin(DATA_DIR, 'fib2.pkl.gz')
    return loads_compat(gzip.open(fname, 'rb').read())


def get_skeleton(name='C1'):
    """ Provide skeletons generated from Local Skeleton Clustering (LSC).

    Parameters
    ----------
    name : str, 'C1' or 'C3'

    Returns
    -------
    dix : dictionary

    Examples
    --------
    >>> from dipy.data import get_skeleton
    >>> C=get_skeleton('C1')
    >>> len(C.keys())
    117
    >>> for c in C: break
    >>> sorted(C[c].keys())
    ['N', 'hidden', 'indices', 'most']

    """
    if name == 'C1':
        fname = pjoin(DATA_DIR, 'C1.pkl.gz')
    if name == 'C3':
        fname = pjoin(DATA_DIR, 'C3.pkl.gz')
    return loads_compat(gzip.open(fname, 'rb').read())


def get_sphere(name='symmetric362'):
    """ provide triangulated spheres

    Parameters
    ----------
    name : str
        which sphere - one of:
        * 'symmetric362'
        * 'symmetric642'
        * 'symmetric724'
        * 'repulsion724'
        * 'repulsion100'
        * 'repulsion200'

    Returns
    -------
    sphere : a dipy.core.sphere.Sphere class instance

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.data import get_sphere
    >>> sphere = get_sphere('symmetric362')
    >>> verts, faces = sphere.vertices, sphere.faces
    >>> verts.shape == (362, 3)
    True
    >>> faces.shape == (720, 3)
    True
    >>> verts, faces = get_sphere('not a sphere name') #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    DataError: No sphere called "not a sphere name"

    """
    fname = SPHERE_FILES.get(name)
    if fname is None:
        raise DataError('No sphere called "%s"' % name)
    res = np.load(fname)
    # Set to native byte order to avoid errors in compiled routines for
    # big-endian platforms, when using these spheres.
    return Sphere(xyz=as_native_array(res['vertices']),
                  faces=as_native_array(res['faces']))


default_sphere = HemiSphere.from_sphere(get_sphere('repulsion724'))
small_sphere = HemiSphere.from_sphere(get_sphere('symmetric362'))


def _gradient_from_file(filename):
    """Reads a gradient file saved as a text file compatible with np.loadtxt
    and saved in the dipy data directory"""
    def gtab_getter():
        gradfile = pjoin(DATA_DIR, filename)
        grad = np.loadtxt(gradfile, delimiter=',')
        gtab = GradientTable(grad)
        return gtab
    return gtab_getter


get_3shell_gtab = _gradient_from_file("gtab_3shell.txt")
get_isbi2013_2shell_gtab = _gradient_from_file("gtab_isbi2013_2shell.txt")
get_gtab_taiwan_dsi = _gradient_from_file("gtab_taiwan_dsi.txt")


def dsi_voxels():
    fimg, fbvals, fbvecs = get_fnames('small_101D')
    bvals = np.loadtxt(fbvals)
    bvecs = np.loadtxt(fbvecs).T
    data, _ = load_nifti(fimg)
    gtab = gradient_table(bvals, bvecs)
    return data, gtab


def dsi_deconv_voxels():
    from dipy.sims.voxel import sticks_and_ball
    gtab = gradient_table(np.loadtxt(get_fnames('dsi515btable')))
    data = np.zeros((2, 2, 2, 515))
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                data[ix, iy, iz], _ = sticks_and_ball(gtab,
                                                      d=0.0015,
                                                      S0=1.,
                                                      angles=[(0, 0),
                                                              (90, 0)],
                                                      fractions=[50, 50],
                                                      snr=None)
    return data, gtab


def mrtrix_spherical_functions():
    """Spherical functions represented by spherical harmonic coefficients and
    evaluated on a discrete sphere.

    Returns
    -------
    func_coef : array (2, 3, 4, 45)
        Functions represented by the coefficients associated with the
        mrtrix spherical harmonic basis of maximal order (l) 8.
    func_discrete : array (2, 3, 4, 81)
        Functions evaluated on `sphere`.
    sphere : Sphere
        The discrete sphere, points on the surface of a unit sphere, used to
        evaluate the functions.

    Notes
    -----
    These coefficients were obtained by using the dwi2SH command of mrtrix.

    """
    func_discrete, _ = load_nifti(pjoin(DATA_DIR, "func_discrete.nii.gz"))
    func_coef, _ = load_nifti(pjoin(DATA_DIR, "func_coef.nii.gz"))
    gradients = np.loadtxt(pjoin(DATA_DIR, "sphere_grad.txt"))
    # gradients[0] and the first volume of func_discrete,
    # func_discrete[..., 0], are associated with the b=0 signal.
    # gradients[:, 3] are the b-values for each gradient/volume.
    sphere = Sphere(xyz=gradients[1:, :3])
    return func_coef, func_discrete[..., 1:], sphere


dipy_cmaps = None


def get_cmap(name):
    """Make a callable, similar to maptlotlib.pyplot.get_cmap."""
    global dipy_cmaps
    if dipy_cmaps is None:
        filename = pjoin(DATA_DIR, "dipy_colormaps.json")
        with open(filename) as f:
            dipy_cmaps = json.load(f)

    desc = dipy_cmaps.get(name)
    if desc is None:
        return None

    def simple_cmap(v):
        """Emulates matplotlib colormap callable"""
        rgba = np.ones((len(v), 4))
        for i, color in enumerate(('red', 'green', 'blue')):
            x, y0, y1 = zip(*desc[color])
            # Matplotlib allows more complex colormaps, but for users who do
            # not have Matplotlib dipy makes a few simple colormaps available.
            # These colormaps are simple because y0 == y1, and therefore we
            # ignore y1 here.
            rgba[:, i] = np.interp(v, x, y0)
        return rgba

    return simple_cmap


def two_cingulum_bundles():
    fname = get_fnames('cb_2')
    res = np.load(fname)
    cb1 = relist_streamlines(res['points'], res['offsets'])
    cb2 = relist_streamlines(res['points2'], res['offsets2'])
    return cb1, cb2


def matlab_life_results():
    matlab_rmse = np.load(pjoin(DATA_DIR, 'life_matlab_rmse.npy'))
    matlab_weights = np.load(pjoin(DATA_DIR, 'life_matlab_weights.npy'))
    return matlab_rmse, matlab_weights


def load_sdp_constraints(model_name, order=None):
    """Import semidefinite programming constraint matrices for different models,
    generated as described for example in [1]_.

    Parameters
    ----------
    model_name : string
        A string identifying the model that is to be constrained.
    order : unsigned int, optional
        A non-negative integer that represent the order or instance of the
        model.
        Default: None.

    Returns
    -------
    sdp_constraints : array
        Constraint matrices

    Notes
    -----
    The constraints will be loaded from a file named 'id_constraint_order.npz'.

    References
    ----------
    .. [1] Dela Haije et al. "Enforcing necessary non-negativity constraints
           for common diffusion MRI models using sum of squares programming".
           NeuroImage 209, 2020, 116405.

    """

    file = model_name + '_constraint'
    if order is not None:
        file += '_' + str(order)
    file += '.npz'
    path = pjoin(DATA_DIR, file)

    if not exists(path):
        raise ValueError("Constraints file '" + file + "' not found.")

    try:
        array = load_npz(path)
        n, x = array.shape
        sdp_constraints = [array[i*x:(i+1)*x] for i in range(n//x)]
        return sdp_constraints
    except Exception:
        raise ValueError("Failed to read constraints file '" + file + "'.")
