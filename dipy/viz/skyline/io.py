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
from dipy.reconst.shm import calculate_max_order
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
SH_BASES = ("descoteaux07", "descoteaux07_legacy", "tournier07", "tournier19")


def _validate_sh_basis(sh_basis):
    """Return a supported input basis, logging invalid declarations.

    Parameters
    ----------
    sh_basis : str
        Input SH convention selected by the caller.

    Returns
    -------
    str
        Supported input convention, or ``"descoteaux07"`` for an invalid
        declaration, which is logged.
    """
    if sh_basis not in SH_BASES:
        logger.error(
            f"sh_basis must be one of {SH_BASES}, got {sh_basis!r}. "
            "Using 'descoteaux07'."
        )
        return "descoteaux07"
    return sh_basis


def _valid_shm_coeffs(coeffs, fname, *, full_basis=False):
    """Check SH volume shape and coefficient count, logging invalid inputs.

    Parameters
    ----------
    coeffs : ndarray or None
        Candidate 4D SH coefficient volume to validate.
    fname : str
        Source filename or display name used in validation errors.
    full_basis : bool, optional
        Whether the coefficient axis includes both even and odd SH orders.

    Returns
    -------
    bool
        True for a 4D array with a valid SH coefficient count, otherwise False.
    """
    if isinstance(coeffs, np.ndarray) and coeffs.ndim == 4:
        try:
            calculate_max_order(coeffs.shape[-1], full_basis=full_basis)
        except ValueError:
            pass
        else:
            return True
    shape = getattr(coeffs, "shape", None)
    coefficient_layout = (
        "full SH coefficient count (1, 4, 9, 16, ...)"
        if full_basis
        else "symmetric SH coefficient count (1, 6, 15, 28, 45, ...)"
    )
    logger.error(
        f"{fname} does not contain SH coefficients: expected a 4D volume with "
        f"a {coefficient_layout}, got shape {shape}."
    )
    return False


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

    Parameters
    ----------
    fname : str
        Path of the NIfTI peaks file.

    Returns
    -------
    tuple or None
        ``(peak_dirs, affine)`` with ``peak_dirs`` of shape (X, Y, Z, N, 3),
        or None if ``fname`` does not hold a recognized peaks layout.

    Notes
    -----
    Accepts the two layouts Skyline recognizes: ``(X, Y, Z, N, 3)``, written
    by ``pam_to_niftis(..., reshape_dirs=False)``, and ``(X, Y, Z, 3*N)``,
    written by ``pam_to_niftis(..., reshape_dirs=True)`` and by MRtrix3
    ``sh2peaks``.
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

    return None


def load_files(
    fnames, *, rois=None, peaks=None, shm_coeffs=None, sh_basis="descoteaux07"
):
    """Load images, peaks, surfaces, and tractograms from ``fnames``.

    Dispatches each path by extension to the matching DIPY loader and
    collects the results into per-type lists. Extensions not recognized in
    ``fnames`` are logged and skipped; ``.npy`` entries are recognized but
    ignored (reserved for BUAN p-value files, not loaded here).

    Parameters
    ----------
    fnames : list of str or None
        Paths to load. Supported extensions: images use ``.nii`` or
        ``.nii.gz``; peaks use ``.pam5``; surfaces use ``.pial``, ``.gii``,
        or ``.gii.gz``; tractograms use ``.trk``, ``.trx``, ``.dpy``,
        ``.tck``, ``.vtk``, ``.vtp``, or ``.fib``.
    rois : list of str, optional
        Paths to ROI images (``.nii`` or ``.nii.gz``); other extensions
        are logged and skipped.
    peaks : list of str, optional
        Paths to PAM or NIfTI peak-direction files.
    shm_coeffs : list of str, optional
        Paths to spherical-harmonic coefficient files (``.pam5``, ``.nii``,
        or ``.nii.gz``); unsupported extensions are logged and skipped.
    sh_basis : str, optional
        Input convention for PAM and NIfTI ODFs: ``"descoteaux07"`` (latest),
        ``"descoteaux07_legacy"`` (old DIPY), ``"tournier07"`` (MRtrix 0.2),
        or ``"tournier19"`` (MRtrix3).

    Returns
    -------
    dict
        Dictionary with keys ``"images"``, ``"peaks"``, ``"rois"``,
        ``"surfaces"``, ``"tractograms"``, ``"shm_coeffs"``, each a list
        of tuples for the matching ``create_*_visualization`` function:

        - images, rois : ``(data, affine, fname)``
        - peaks : ``(peak_dirs, affine, fname, peak_values)``; NIfTI files
          have ``peak_values=None``.
        - surfaces : ``(vertices, faces, fname)``
        - tractograms : ``(sft, fname)``
        - shm_coeffs : ``(coeffs, affine, fname, sh_basis)``

    Notes
    -----
    SH coefficients remain in the declared input basis; only rendering converts
    them. The convention is not inferred. Unsupported names are logged and use
    ``"descoteaux07"``.
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
        if ext in [".nii.gz", ".nii", ".pam5"]:
            if ext == ".pam5":
                pam = load_pam(fname)
                data, affine = pam.shm_coeff, pam.affine
            elif ext in [".nii.gz", ".nii"]:
                data, affine = load_nifti(fname)
            if _valid_shm_coeffs(data, fname):
                input_basis = _validate_sh_basis(sh_basis)
                skyline_shm_coeffs.append((data, affine, fname, input_basis))
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
    ndarray or None
        The loaded array, or None if the file could not be loaded.
    """
    try:
        data = np.load(fname)
        return data
    except (OSError, ValueError, EOFError) as e:
        logger.error(f"Error loading numpy file '{fname}': {e}")
        return None
