"""Load mixed media files for Skyline from disk paths.

``EMERGENCY_REF`` supplies a fallback NIfTI header (MNI-like spacing) when
tractograms must load before any matching reference image is available.
"""

import numpy as np

from dipy.io.image import load_nifti
from dipy.io.peaks import load_pam
from dipy.io.streamline import load_tractogram
from dipy.io.surface import load_gifti, load_pial
from dipy.io.utils import create_nifti_header, split_filename_extension
from dipy.reconst.shm import calculate_max_order, convert_sh_descoteaux_tournier
from dipy.utils.logging import logger

mni_2009c = {
    "affine": np.array(
        [
            [1.0, 0.0, 0.0, -96.0],
            [0.0, 1.0, 0.0, -132.0],
            [0.0, 0.0, 1.0, -78.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "dims": (193, 229, 193),
    "vox_size": (1.0, 1.0, 1.0),
    "vox_space": "RAS",
}
EMERGENCY_REF = create_nifti_header(
    mni_2009c["affine"], mni_2009c["dims"], mni_2009c["vox_size"]
)
SH_BASES = ("descoteaux07", "tournier07")


def _reference_from_image(data, affine):
    """Build a NIfTI header usable as a tractogram spatial reference.

    Formats without an embedded header (``.tck``, ``.vtk``, ``.dpy``, ...) need
    a full reference, not just an affine.

    Parameters
    ----------
    data : ndarray
        Volume the reference geometry is taken from.
    affine : ndarray, shape (4, 4)
        Voxel-to-world transform of ``data``.

    Returns
    -------
    nibabel.nifti1.Nifti1Header
        Header carrying the volume's affine, dimensions and voxel sizes.
    """
    vox_size = np.linalg.norm(affine[:3, :3], axis=0)
    return create_nifti_header(affine, data.shape[:3], vox_size)


def _peaks_from_nifti(fname):
    """Load peak directions from a NIfTI peaks volume.

    Accepts the two layouts Skyline recognizes: ``(X, Y, Z, N, 3)``, written
    by ``pam_to_niftis(..., reshape_dirs=False)``, and ``(X, Y, Z, 3*N)``,
    written by ``pam_to_niftis(..., reshape_dirs=True)`` and by MRtrix3
    ``sh2peaks``.

    Parameters
    ----------
    fname : str
        Path of the NIfTI peaks file.

    Returns
    -------
    tuple or None
        ``(peak_dirs, affine)`` with ``peak_dirs`` of shape (X, Y, Z, N, 3),
        or None if ``fname`` does not hold a recognized peaks layout.
    """
    data, affine = load_nifti(fname)
    if data.ndim == 4 and data.shape[-1] % 3 == 0:
        data = data.reshape(data.shape[:3] + (-1, 3))
    elif data.ndim != 5 or data.shape[-1] != 3:
        logger.error(
            f"{fname} is not a peaks volume: expected shape (X, Y, Z, N, 3) or "
            f"(X, Y, Z, 3*N), got {data.shape}."
        )
        return None
    return np.nan_to_num(data.astype(np.float32), copy=False), affine


def _shm_from_nifti(fname, sh_basis):
    """Load SH coefficients from a 4D NIfTI ODF volume.

    Parameters
    ----------
    fname : str
        Path of the NIfTI SH coefficients file.
    sh_basis : str
        SH basis of ``fname``: ``"descoteaux07"`` or ``"tournier07"``. Any
        value other than ``"tournier07"`` is treated as ``"descoteaux07"``.

    Returns
    -------
    tuple or None
        ``(coeffs, affine)`` with ``coeffs`` converted to legacy
        descoteaux07, or None if ``fname`` does not hold a symmetric SH
        coefficient volume.
    """
    data, affine = load_nifti(fname)
    if data.ndim == 4:
        try:
            calculate_max_order(data.shape[-1])
        except ValueError:
            pass
        else:
            coeffs = data.astype(np.float32, copy=False)
            if sh_basis == "tournier07":
                coeffs = convert_sh_descoteaux_tournier(coeffs)
            return coeffs, affine
    logger.error(
        f"{fname} does not contain SH coefficients: expected a 4D volume with "
        f"a symmetric SH coefficient count (1, 6, 15, 28, 45, ...), got shape "
        f"{data.shape}."
    )
    return None


def load_files(
    fnames, *, rois=None, peaks=None, shm_coeffs=None, sh_basis="descoteaux07"
):
    """Load the provided list of files.

    Parameters
    ----------
    fnames : list of str
        Path of the file.
    rois : list of str, optional
        Paths of the ROIs.
    peaks : list of str, optional
        Paths of the peak files.
    shm_coeffs : list of str, optional
        Paths of the SH coefficients files.
    sh_basis : str, optional
        SH basis of NIfTI ODFs in ``shm_coeffs``: ``"descoteaux07"`` or
        ``"tournier07"``.

    Returns
    -------
    dict
        Dictionary containing the loaded images, peaks, ROIs, surfaces,
        tractograms, and spherical-harmonic coefficient data.

    Notes
    -----
    NIfTI peak files (``.nii``, ``.nii.gz``) are accepted in two layouts:
    ``(X, Y, Z, N, 3)``, as written by
    ``dipy.io.peaks.pam_to_niftis(..., reshape_dirs=False)``, and
    ``(X, Y, Z, 3*N)``, as written with ``reshape_dirs=True`` and by MRtrix3
    ``sh2peaks``. Each ``"peaks"`` entry is a
    ``(peak_dirs, affine, filename, peak_values)`` tuple; ``peak_values`` is
    None for NIfTI peaks, since a NIfTI peaks file carries no magnitude
    information.

    NIfTI ODF files must be a 4D volume whose last dimension is a symmetric
    SH coefficient count (1, 6, 15, 28, 45, ...). Coefficients in the
    ``tournier07`` (MRtrix3) basis are converted to legacy ``descoteaux07``.
    """
    if fnames is None:
        fnames = []

    if rois is None:
        rois = []

    if peaks is None:
        peaks = []

    if shm_coeffs is None:
        shm_coeffs = []

    skyline_images = []
    skyline_peaks = []
    skyline_rois = []
    skyline_surfaces = []
    skyline_tractograms = []
    skyline_shm_coeffs = []

    for fname in fnames:
        logger.info(f"Loading file ... \n{fname}\n")
        _, ext = split_filename_extension(fname)
        ext = ext.lower()

        if ext in [".nii.gz", ".nii"]:
            data, affine = load_nifti(fname)
            skyline_images.append((data, affine, fname))
        elif ext == ".pam5":
            pam = load_pam(fname)
            skyline_peaks.append((pam.peak_dirs, pam.affine, fname, pam.peak_values))
        elif ext == ".pial":
            surface = load_pial(fname)
            if surface:
                vertices, faces = surface
                skyline_surfaces.append((vertices, faces, fname))
        elif any(ext.endswith(_ext) for _ext in [".gii", ".gii.gz"]):
            surface = load_gifti(fname)
            vertices, faces = surface
            if len(vertices) and len(faces):
                vertices, faces = surface
                skyline_surfaces.append((vertices, faces, fname))
            else:
                logger.warning(
                    f"{fname} does not have any surface geometry.", stacklevel=2
                )
        elif ext in [".trk", ".trx"]:
            sft = load_tractogram(fname, "same", bbox_valid_check=False)
            skyline_tractograms.append((sft, fname))
        elif ext in [".dpy", ".tck", ".vtk", ".vtp", ".fib"]:
            if skyline_images:
                sft = load_tractogram(
                    fname,
                    _reference_from_image(*skyline_images[0][:2]),
                    bbox_valid_check=False,
                )
            else:
                sft = load_tractogram(fname, EMERGENCY_REF)
            skyline_tractograms.append((sft, fname))
        elif ext == ".npy":
            # To support horizon BUAN p-values file
            pass
        else:
            logger.error(f"File extension '{ext}' is not supported in Skyline.")

    for fname in rois:
        logger.info(f"Loading file ... \n{fname}\n")
        _, ext = split_filename_extension(fname)
        ext = ext.lower()
        if ext in [".nii.gz", ".nii"]:
            data, affine = load_nifti(fname)
            skyline_rois.append((data, affine, fname))
        else:
            logger.error(
                f"File extension '{ext}' is not supported for ROIs in Skyline."
            )

    for fname in peaks:
        logger.info(f"Loading file ... \n{fname}\n")
        _, ext = split_filename_extension(fname)
        ext = ext.lower()
        if ext == ".pam5":
            pam = load_pam(fname)
            skyline_peaks.append((pam.peak_dirs, pam.affine, fname, pam.peak_values))
        elif ext in [".nii.gz", ".nii"]:
            result = _peaks_from_nifti(fname)
            if result is not None:
                skyline_peaks.append((*result, fname, None))
        else:
            logger.error(
                f"File extension '{ext}' is not supported for peaks in Skyline."
            )

    for fname in shm_coeffs:
        logger.info(f"Loading file ... \n{fname}\n")
        _, ext = split_filename_extension(fname)
        ext = ext.lower()
        if ext == ".pam5":
            pam = load_pam(fname)
            skyline_shm_coeffs.append((pam.shm_coeff, pam.affine, fname, "descoteaux"))
        elif ext in [".nii.gz", ".nii"]:
            result = _shm_from_nifti(fname, sh_basis)
            if result is not None:
                skyline_shm_coeffs.append((*result, fname, "descoteaux"))
        else:
            logger.error(
                f"File extension '{ext}' is not supported for ODFs in Skyline."
            )

    return {
        "images": skyline_images,
        "peaks": skyline_peaks,
        "rois": skyline_rois,
        "surfaces": skyline_surfaces,
        "tractograms": skyline_tractograms,
        "shm_coeffs": skyline_shm_coeffs,
    }


def load_npy(fname):
    """Load a numpy file containing BUAN color values.

    Parameters
    ----------
    fname : str
        Path to the .npy file.

    Returns
    -------
    np.ndarray
        The loaded numpy array.
    """
    try:
        data = np.load(fname)
        return data
    except Exception as e:
        logger.error(f"Error loading numpy file '{fname}': {e}")
        return None
