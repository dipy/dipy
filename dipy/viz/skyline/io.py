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


def load_files(fnames, *, rois=None, shm_coeffs=None):
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
    shm_coeffs : list of str, optional
        Paths to spherical-harmonic coefficient files (``.pam5``); other
        extensions are silently skipped.

    Returns
    -------
    dict
        Dictionary with keys ``"images"``, ``"peaks"``, ``"rois"``,
        ``"surfaces"``, ``"tractograms"``, ``"shm_coeffs"``, each a list
        of tuples for the matching ``create_*_visualization`` function:

        - images, rois : ``(data, affine, fname)``
        - peaks : ``(pam, fname)``
        - surfaces : ``(vertices, faces, fname)``
        - tractograms : ``(sft, fname)``
        - shm_coeffs : ``(coeffs, affine, fname, "descoteaux")``
    """
    if fnames is None:
        fnames = []

    if rois is None:
        rois = []

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
            skyline_peaks.append((pam, fname))
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

    for fname in shm_coeffs:
        logger.info(f"Loading file ... \n{fname}\n")
        _, ext = split_filename_extension(fname)
        ext = ext.lower()
        if ext == ".pam5":
            pam = load_pam(fname)
            skyline_shm_coeffs.append((pam.shm_coeff, pam.affine, fname, "descoteaux"))

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
    except Exception as e:
        logger.error(f"Error loading numpy file '{fname}': {e}")
        return None
