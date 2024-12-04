import logging
import os
import time
from warnings import warn

import nibabel as nib
import numpy as np
import vtk
import vtk.util.numpy_support as ns

from dipy.io.vtk import load_polydata, save_polydata
from dipy.io.stateful_surface import StatefulSurface, Origin, Space
from dipy.io.utils import is_header_compatible, get_reference_info
from dipy.testing.decorators import warning_for_keywords


def load_surface(
    fname,
    reference,
    *,
    to_space=Space.RASMM,
    to_origin=Origin.NIFTI,
    bbox_valid_check=True,
    gii_header_check=True,
    from_space=None,
    from_origin=None,
):
    """Load the stateful surface from any format (vtk/vtp/obj/stl/ply/gii/pial)

    Parameters
    ----------
    filename : string
        Filename with valid extension
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict), or 'same' if the input is a trk file.
        Reference that provides the spatial attribute.
        Typically a nifti-related object from the native diffusion used for
        streamlines generation
    to_space : Enum (dipy.io.stateful_surface.Space)
        Space to which the surface will be transformed after loading
    to_origin : Enum (dipy.io.stateful_surface.Origin)
        Origin to which the surface will be transformed after loading
            NIFTI standard, default (center of the voxel)
            TRACKVIS standard (corner of the voxel)
    bbox_valid_check : bool
        Verification for negative voxel coordinates or values above the
        volume dimensions. Default is True, to enforce valid file.
    gii_header_check : bool
        Verification that the reference has the same header as the spatial
        attributes as the input tractogram when a GII is loaded
    from_space : Enum (dipy.io.stateful_surface.Space)
        Space to which the surface was transformed before saving.
        Help for software compatibility. If None, assumes RASMM.
    from_origin : Enum (dipy.io.stateful_surface.Origin)
        Origin to which the surface was transformed before saving.
        Help for software compatibility. If None, assumes NIFTI.

    Returns
    -------
    output : StatefulSurface
        The surface to load (must have been saved properly)
    """
    _, extension = os.path.splitext(fname)
    if extension not in [".vtk", ".vtp", ".obj", ".stl", ".ply", ".gii", ".gii.gz",
                         ".pial"]:
        logging.error("Output filename is not one of the supported format.")
        return False

    if to_space not in Space:
        logging.error("Space MUST be one of the 3 choices (Enum).")
        return False

    if reference == "same":
        if extension in [".gii", ".gii.gz"]:
            reference = fname
        else:
            logging.error(
                'Reference must be provided, "same" is only ' "available for GII file."
            )
            return False

    if gii_header_check and extension in [".gii", ".gii.gz"]:
        if not is_header_compatible(fname, reference):
            logging.error(
                "Trk file header does not match the provided reference.")
            return False
    _, ext = os.path.splitext(fname)

    timer = time.time()
    if ext == ".gii" or ext == ".gii.gz":
        data = load_gifti(fname)
    elif ext in [".vtk", ".vtp", ".obj", ".stl", ".ply"]:
        data = load_polydata(fname)
    else:
        try:
            data = load_pial(fname)
            reference = fname if reference == "same" else reference
            affine, dimensions, _, _ = get_reference_info(reference)
            center_volume = (np.array(dimensions) / 2)
            xform_translation = np.dot(affine[0:3, 0:3], center_volume) + affine[0:3, 3]
            data = data[0] + xform_translation, data[1]

            if from_space is not None or from_origin is not None:
                warn("from_space and from_origin are ignored when loading pial files.")
            from_space = Space.RASMM
            from_origin = Origin.NIFTI
        except ValueError:
            warn(f"The file {fname} provided is not supported.")

    from_space = Space.RASMM if from_space is None else from_space
    from_origin = Origin.NIFTI if from_origin is None else from_origin

    sfs = StatefulSurface(data, reference, space=from_space, origin=from_origin,
                          data_per_point=None)
    logging.debug(
        "Load %s with %s streamlines in %s seconds.",
        fname,
        len(sfs),
        round(time.time() - timer, 3),
    )

    if bbox_valid_check and not sfs.is_bbox_in_vox_valid():
        raise ValueError(
            "Bounding box is not valid in voxel space, cannot "
            "load a valid file if some coordinates are invalid.\n"
            "Please set bbox_valid_check to False and then use "
            "the function remove_invalid_streamlines to discard "
            "invalid streamlines."
        )

    sfs.to_space(to_space)
    sfs.to_origin(to_origin)

    return sfs


def save_surface(fname, sfs, to_space=Space.RASMM, to_origin=Origin.NIFTI,
                 legacy_vtk_format=False):
    """
    """
    sfs.to_space(to_space)
    sfs.to_origin(to_origin)
    save_polydata(sfs.get_polydata(), fname, legacy_vtk_format=legacy_vtk_format)



@warning_for_keywords()
def load_pial(fname, *, return_meta=False):
    """Load pial file.

    Parameters
    ----------
    fname : str
        Absolute path of the file.
    return_meta : bool, optional
        Whether to read the metadata of the file or not, by default False.

    Returns
    -------
    tuple
        (vertices, faces) if return_meta=False. Otherwise, (vertices, faces,
        metadata).
    """
    try:
        return nib.freesurfer.read_geometry(fname, read_metadata=return_meta)
    except ValueError:
        warn(
            f"The file {fname} provided does not have geometry data.", stacklevel=2)


def load_gifti(fname):
    """Load gifti file.

    Parameters
    ----------
    fname : str
        Absolute path of the file.

    Returns
    -------
    tuple
        (vertices, faces)
    """
    surf_img = nib.load(fname)
    return surf_img.agg_data(("pointset", "triangle"))
