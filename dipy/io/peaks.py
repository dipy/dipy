import os

import h5py
import numpy as np

from dipy.core.sphere import Sphere
from dipy.direction.peaks import PeaksAndMetrics, reshape_peaks_for_visualization
from dipy.io.image import save_nifti
from dipy.reconst.dti import quantize_evecs
from dipy.testing.decorators import warning_for_keywords
from dipy.utils.deprecator import deprecate_with_version


def _safe_save(group, array, name):
    """Safe saving of arrays with specific names.

    Parameters
    ----------
    group : HDF5 group
    array : array
    name : string

    """
    if array is not None:
        ds = group.create_dataset(
            name, shape=array.shape, dtype=array.dtype, chunks=True
        )
        ds[:] = array


@deprecate_with_version(
    "dipy.io.peaks.load_peaks is deprecated, Please use"
    "dipy.io.peaks.load_pam instead",
    since="1.10",
    until="1.12",
)
@warning_for_keywords()
def load_peaks(fname, *, verbose=False):
    """Load a PeaksAndMetrics HDF5 file (PAM5).

    Parameters
    ----------
    fname : string
        Filename of PAM5 file.
    verbose : bool
        Print summary information about the loaded file.

    Returns
    -------
    pam : PeaksAndMetrics object

    """
    return load_pam(fname=fname, verbose=verbose)


def load_pam(fname, *, verbose=False):
    """Load a PeaksAndMetrics HDF5 file (PAM5).

    Parameters
    ----------
    fname : string
        Filename of PAM5 file.
    verbose : bool, optional
        Print summary information about the loaded file.

    Returns
    -------
    pam : PeaksAndMetrics object
        Object holding peaks information and metrics.

    """
    if os.path.splitext(fname)[1].lower() != ".pam5":
        raise IOError("This function supports only PAM5 (HDF5) files")

    f = h5py.File(fname, "r")

    pam = PeaksAndMetrics()

    pamh = f["pam"]

    version = f.attrs["version"]

    if version != "0.0.1":
        raise OSError(f"Incorrect PAM5 file version {version}")

    peak_dirs = pamh["peak_dirs"][:]
    peak_values = pamh["peak_values"][:]
    peak_indices = pamh["peak_indices"][:]

    sphere_vertices = pamh["sphere_vertices"][:]

    pam.affine = pamh["affine"][:] if "affine" in pamh else None
    pam.peak_dirs = peak_dirs
    pam.peak_values = peak_values
    pam.peak_indices = peak_indices
    pam.shm_coeff = pamh["shm_coeff"][:] if "shm_coeff" in pamh else None
    pam.sphere = Sphere(xyz=sphere_vertices)
    pam.B = pamh["B"][:] if "B" in pamh else None
    pam.total_weight = pamh["total_weight"][:][0] if "total_weight" in pamh else None
    pam.ang_thr = pamh["ang_thr"][:][0] if "ang_thr" in pamh else None
    pam.gfa = pamh["gfa"][:] if "gfa" in pamh else None
    pam.qa = pamh["qa"][:] if "qa" in pamh else None
    pam.odf = pamh["odf"][:] if "odf" in pamh else None

    f.close()

    if verbose:
        print("PAM5 version")
        print(version)
        print("Affine")
        print(pam.affine)
        print("Dirs shape")
        print(pam.peak_dirs.shape)
        print("SH shape")
        if pam.shm_coeff is not None:
            print(pam.shm_coeff.shape)
        else:
            print("None")
        print("ODF shape")
        if pam.odf is not None:
            print(pam.odf.shape)
        else:
            print("None")
        print("Total weight")
        print(pam.total_weight)
        print("Angular threshold")
        print(pam.ang_thr)
        print("Sphere vertices shape")
        print(pam.sphere.vertices.shape)

    return pam


@deprecate_with_version(
    "dipy.io.peaks.save_peaks is deprecated, Please use "
    "dipy.io.peaks.save_pam instead",
    since="1.10.0",
    until="1.12.0",
)
@warning_for_keywords()
def save_peaks(fname, pam, *, affine=None, verbose=False):
    """Save PeaksAndMetrics object attributes in a PAM5 file (HDF5).

    Parameters
    ----------
    fname : string
        Filename of PAM5 file.
    pam : PeaksAndMetrics
        Object holding peak_dirs, shm_coeffs and other attributes.
    affine : array
        The 4x4 matrix transforming the date from native to world coordinates.
        PeaksAndMetrics should have that attribute but if not it can be
        provided here. Default None.
    verbose : bool
        Print summary information about the saved file.

    """
    return save_pam(fname=fname, pam=pam, affine=affine, verbose=verbose)


def save_pam(fname, pam, *, affine=None, verbose=False):
    """Save all important attributes of object PeaksAndMetrics in a PAM5 file (HDF5).

    Parameters
    ----------
    fname : str
        Filename of PAM5 file.
    pam : PeaksAndMetrics
        Object holding peaks information and metrics.
    affine : ndarray, optional
        The 4x4 matrix transforming the date from native to world coordinates.
        PeaksAndMetrics should have that attribute but if not it can be
        provided here.
    verbose : bool, optional
        Print summary information about the saved file.

    """
    if os.path.splitext(fname)[1] != ".pam5":
        raise IOError("This function saves only PAM5 (HDF5) files")

    if not (
        hasattr(pam, "peak_dirs")
        and hasattr(pam, "peak_values")
        and hasattr(pam, "peak_indices")
    ):
        msg = "Cannot save object without peak_dirs, peak_values"
        msg += " and peak_indices"
        raise ValueError(msg)

    if not (
        isinstance(pam.peak_dirs, np.ndarray)
        and isinstance(pam.peak_values, np.ndarray)
        and isinstance(pam.peak_indices, np.ndarray)
    ):
        msg = "Cannot save object: peak_dirs, peak_values"
        msg += " and peak_indices should be a ndarray"
        raise ValueError(msg)

    f = h5py.File(fname, "w")

    group = f.create_group("pam")
    f.attrs["version"] = "0.0.1"

    version_string = f.attrs["version"]

    affine = pam.affine if hasattr(pam, "affine") else affine
    shm_coeff = pam.shm_coeff if hasattr(pam, "shm_coeff") else None
    odf = pam.odf if hasattr(pam, "odf") else None
    vertices = None
    if hasattr(pam, "sphere") and pam.sphere is not None:
        vertices = pam.sphere.vertices

    _safe_save(group, affine, "affine")
    _safe_save(group, pam.peak_dirs, "peak_dirs")
    _safe_save(group, pam.peak_values, "peak_values")
    _safe_save(group, pam.peak_indices, "peak_indices")
    _safe_save(group, shm_coeff, "shm_coeff")
    _safe_save(group, vertices, "sphere_vertices")
    _safe_save(group, pam.B, "B")
    _safe_save(group, np.array([pam.total_weight]), "total_weight")
    _safe_save(group, np.array([pam.ang_thr]), "ang_thr")
    _safe_save(group, pam.gfa, "gfa")
    _safe_save(group, pam.qa, "qa")
    _safe_save(group, odf, "odf")

    f.close()

    if verbose:
        print("PAM5 version")
        print(version_string)
        print("Affine")
        print(affine)
        print("Dirs shape")
        print(pam.peak_dirs.shape)
        print("SH shape")
        if shm_coeff is not None:
            print(shm_coeff.shape)
        else:
            print("None")
        print("ODF shape")
        if odf is not None:
            print(pam.odf.shape)
        else:
            print("None")
        print("Total weight")
        print(pam.total_weight)
        print("Angular threshold")
        print(pam.ang_thr)
        print("Sphere vertices shape")
        print(pam.sphere.vertices.shape)

    return pam


@deprecate_with_version(
    "dipy.io.peaks.peaks_to_niftis is deprecated, Please"
    " use dipy.io.peaks.pam_to_niftis instead",
    since="1.10.0",
    until="1.12.0",
)
@warning_for_keywords()
def peaks_to_niftis(
    pam,
    fname_shm,
    fname_dirs,
    fname_values,
    fname_indices,
    *,
    fname_gfa=None,
    reshape_dirs=False,
):
    """Save SH, directions, indices and values of peaks to Nifti.

    Parameters
    ----------
    pam : PeaksAndMetrics
        Object holding peaks information and metrics.
    fname_shm : str
        Spherical Harmonics coefficients filename.
    fname_dirs : str
        Peaks direction filename.
    fname_values : str
        Peaks values filename.
    fname_indices : str
        Peaks indices filename.
    fname_gfa : str, optional
        Generalized FA filename.
    reshape_dirs : bool, optional
        If True, reshape peaks for visualization.

    """
    return pam_to_niftis(
        pam=pam,
        fname_peaks_dir=fname_dirs,
        fname_peaks_values=fname_values,
        fname_peaks_indices=fname_indices,
        fname_gfa=fname_gfa,
        fname_shm=fname_shm,
        reshape_dirs=reshape_dirs,
    )


def pam_to_niftis(
    pam,
    *,
    fname_peaks_dir="peaks_dirs.nii.gz",
    fname_peaks_values="peaks_values.nii.gz",
    fname_peaks_indices="peaks_indices.nii.gz",
    fname_shm="shm.nii.gz",
    fname_gfa="gfa.nii.gz",
    fname_sphere="sphere.txt",
    fname_b="B.nii.gz",
    fname_qa="qa.nii.gz",
    reshape_dirs=False,
):
    """Save SH, directions, indices and values of peaks to Nifti.

    Parameters
    ----------
    pam : PeaksAndMetrics
        Object holding peaks information and metrics.
    fname_peaks_dir : str, optional
        Peaks direction filename.
    fname_peaks_values : str, optional
        Peaks values filename.
    fname_peaks_indices : str, optional
        Peaks indices filename.
    fname_shm : str, optional
        Spherical Harmonics coefficients filename. It will be saved if available.
    fname_gfa : str, optional
        Generalized FA filename. It will be saved if available.
    fname_sphere : str, optional
        Sphere vertices filename. It will be saved if available.
    fname_b : str, optional
        B Matrix filename. Matrix that transforms spherical harmonics to
        spherical function. It will be saved if available.
    fname_qa : str, optional
        Quantitative Anisotropy filename. It will be saved if available.
    reshape_dirs : bool, optional
        If True, Reshape and convert to float32 a set of peaks for
        visualisation with mrtrix or the fibernavigator.

    """
    if reshape_dirs:
        pam_dirs = reshape_peaks_for_visualization(pam)
    else:
        pam_dirs = pam.peak_dirs.astype(np.float32)

    save_nifti(fname_peaks_dir, pam_dirs, pam.affine)
    save_nifti(fname_peaks_values, pam.peak_values.astype(np.float32), pam.affine)
    save_nifti(fname_peaks_indices, pam.peak_indices, pam.affine)

    for attr, fname in [("gfa", fname_gfa), ("B", fname_b), ("qa", fname_qa)]:
        obj = getattr(pam, attr, None)
        if obj is None:
            continue
        save_nifti(fname, obj, pam.affine)

    if hasattr(pam, "shm_coeff") and pam.shm_coeff is not None:
        save_nifti(fname_shm, pam.shm_coeff.astype(np.float32), pam.affine)
    if hasattr(pam, "sphere") and pam.sphere is not None:
        np.savetxt(fname_sphere, pam.sphere.vertices)


def niftis_to_pam(
    affine,
    peak_dirs,
    peak_values,
    peak_indices,
    *,
    shm_coeff=None,
    sphere=None,
    gfa=None,
    B=None,
    qa=None,
    odf=None,
    total_weight=None,
    ang_thr=None,
    pam_file=None,
):
    """Return SH, directions, indices and values of peaks to pam5.

    Parameters
    ----------
    affine : array, (4, 4)
        The matrix defining the affine transform.
    peak_dirs : ndarray
        The direction of each peak.
    peak_values : ndarray
        The value of the peaks.
    peak_indices : ndarray
        Indices (in sphere vertices) of the peaks in each voxel.
    shm_coeff : array, optional
        Spherical harmonics coefficients.
    sphere : `Sphere` class instance, optional
        The sphere providing discrete directions for evaluation.
    gfa : ndarray, optional
        Generalized FA volume.
    B : ndarray, optional
        Matrix that transforms spherical harmonics to spherical function.
    qa : array, optional
        Quantitative Anisotropy in each voxel.
    odf : ndarray, optional
        SH coefficients for the ODF spherical function.
    total_weight : float, optional
        Total weight of the peaks.
    ang_thr : float, optional
        Angular threshold of the peaks.
    pam_file : str, optional
        Filename of the desired pam file.

    Returns
    -------
    pam : PeaksAndMetrics
        Object holding peak_dirs, shm_coeffs and other attributes.

    """
    pam = PeaksAndMetrics()
    pam.affine = affine
    pam.peak_dirs = peak_dirs
    pam.peak_values = peak_values
    pam.peak_indices = peak_indices

    for name, value in [
        ("shm_coeff", shm_coeff),
        ("sphere", sphere),
        ("B", B),
        ("total_weight", total_weight),
        ("ang_thr", ang_thr),
        ("gfa", gfa),
        ("qa", qa),
        ("odf", odf),
    ]:
        if value is not None:
            setattr(pam, name, value)

    if pam_file:
        save_pam(pam_file, pam)
    return pam


def tensor_to_pam(
    evals,
    evecs,
    affine,
    *,
    shm_coeff=None,
    sphere=None,
    gfa=None,
    B=None,
    qa=None,
    odf=None,
    total_weight=None,
    ang_thr=None,
    pam_file=None,
    npeaks=5,
    generate_peaks_indices=True,
):
    """Convert diffusion tensor to pam5.

    Parameters
    ----------
    evals : ndarray
        Eigenvalues of a diffusion tensor. shape should be (...,3).
    evecs : ndarray
        Eigen vectors from the tensor model.
    affine : array, (4, 4)
        The matrix defining the affine transform.
    shm_coeff : array, optional
        Spherical harmonics coefficients.
    sphere : `Sphere` class instance, optional
        The sphere providing discrete directions for evaluation.
    gfa : ndarray, optional
        Generalized FA volume.
    B : ndarray, optional
        Matrix that transforms spherical harmonics to spherical function.
    qa : array, optional
        Quantitative Anisotropy in each voxel.
    odf : ndarray, optional
        SH coefficients for the ODF spherical function.
    pam_file : str, optional
        Filename of the desired pam file.
    npeaks : int, optional
        Maximum number of peaks found.
    generate_peaks_indices : bool, optional
    total_weight : float, optional
        Total weight of the peaks.
    ang_thr : float, optional
        Angular threshold of the peaks.

    Returns
    -------
    pam : PeaksAndMetrics
        Object holding peaks information and metrics.
    """
    npeaks = 1 if npeaks < 1 else npeaks
    npeaks = min(npeaks, evals.shape[-1])
    shape = evals.shape[:3]
    peaks_dirs = np.zeros((shape + (npeaks, 3)))
    peaks_dirs[..., :npeaks, :] = evecs[..., :npeaks, :]
    peaks_values = np.zeros((shape + (npeaks,)))
    peaks_values[..., :npeaks] = evals[..., :npeaks]

    if generate_peaks_indices:
        vertices = sphere.vertices if sphere else None
        peaks_indices = quantize_evecs(evecs[..., :npeaks, :], odf_vertices=vertices)
    else:
        peaks_indices = np.zeros((shape + (npeaks,)), dtype="int")
        peaks_indices.fill(-1)

    return niftis_to_pam(
        affine=affine,
        peak_dirs=peaks_dirs,
        peak_values=peaks_values,
        peak_indices=peaks_indices.astype(np.int32),
        shm_coeff=shm_coeff,
        sphere=sphere,
        gfa=gfa,
        B=B,
        qa=qa,
        odf=odf,
        total_weight=total_weight,
        ang_thr=ang_thr,
        pam_file=pam_file,
    )
