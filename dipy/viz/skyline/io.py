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


def load_files(fnames, rois=None):
    """Load the provided list of files.

    Parameters
    ----------
    fnames : list of str
        Path of the file.
    rois : list of str, optional
        Paths of the ROIs.

    Returns
    -------
    dict
         A dictionary containing the loaded data in the following keys:
        - "images": List of tuples (data, affine, filename) for each image file.
        - "peaks": List of tuples (pam, filename) for each peak file.
        - "rois": List of tuples (data, affine, filename) for each ROI.
        - "surfaces": List of tuples (vertices, faces, filename) for each surface file.
        - "tractograms": List of tuples (sft, filename) for each tractogram file.
    """
    if fnames is None:
        fnames = []

    if rois is None:
        rois = []

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
            if hasattr(pam, "shm_coeff"):
                skyline_shm_coeffs.append(
                    (pam.shm_coeff, pam.affine, fname, "descoteaux")
                )
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
                    fname, skyline_images[0][1], bbox_valid_check=False
                )
            else:
                sft = load_tractogram(fname, EMERGENCY_REF)
            skyline_tractograms.append((sft, fname))
        else:
            logger.error(f"File extension '{ext}' is not supported in Skyline.")

    for fname in rois:
        _, ext = split_filename_extension(fname)
        ext = ext.lower()
        if ext in [".nii.gz", ".nii"]:
            data, affine = load_nifti(fname)
            skyline_rois.append((data, affine, fname))
        else:
            logger.error(
                f"File extension '{ext}' is not supported for ROIs in Skyline."
            )

    return {
        "images": skyline_images,
        "peaks": skyline_peaks,
        "rois": skyline_rois,
        "surfaces": skyline_surfaces,
        "tractograms": skyline_tractograms,
        "shm_coeffs": skyline_shm_coeffs,
    }
