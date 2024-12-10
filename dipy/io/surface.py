from copy import deepcopy
import logging
import os
import time
from warnings import warn

import nibabel as nib
import numpy as np
import vtk
import vtk.util.numpy_support as ns

from dipy.io.vtk import load_polydata, save_polydata
from dipy.io.stateful_surface import StatefulSurface
from dipy.io.utils import is_header_compatible, get_reference_info
from dipy.testing.decorators import warning_for_keywords
from dipy.io.utils import Origin, Space


def load_surface(
    fname,
    reference,
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

    Raises
    ------
    ValueError
        If the bounding box is not valid in voxel space.

    Returns
    -------
    output : StatefulSurface
        The surface to load (must have been saved properly)
    """
    vtk_ext = [".vtk", ".vtp", ".obj", ".stl", ".ply"]
    freesurfer_ext = [".gii", ".gii.gz", ".pial", ".nofix", ".orig",
                      ".smoothwm", ".T1"]
    _, extension = os.path.splitext(fname)
    if extension not in freesurfer_ext + vtk_ext:
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
    metadata = None
    if ext == ".gii" or ext == ".gii.gz":
        data = load_gifti(fname)
    elif ext in [".vtk", ".vtp", ".obj", ".stl", ".ply"]:
        data = load_polydata(fname)
    else:
        data = load_pial(fname, return_meta=True)
        data, metadata = data[0:2], data[2]
        print('==++==', data[0][0])

        reference = fname if reference == "same" else reference
        affine, dimensions, _, _ = get_reference_info(reference)
        center_volume = (np.array(dimensions) / 2)
        xform_translation = np.dot(
            affine[0:3, 0:3], center_volume - 0.5) + affine[0:3, 3]

        data = data[0] + xform_translation, data[1]

        if from_space is not None or from_origin is not None:
            logging.warning(
                "from_space and from_origin are ignored when loading pial files.")
        from_space = Space.RASMM
        from_origin = Origin.NIFTI

    from_space = Space.RASMM if from_space is None else from_space
    from_origin = Origin.NIFTI if from_origin is None else from_origin
    sfs = StatefulSurface(data, reference, space=from_space, origin=from_origin,
                          data_per_point=None)
    print('====', sfs.vertices[0])
    sfs.metadata = metadata

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
            "invalid streamlines.")

    sfs.to_space(to_space)
    sfs.to_origin(to_origin)
    print('==++==', sfs.vertices[0])

    return sfs


def save_surface(fname, sfs, to_space=Space.RASMM, to_origin=Origin.NIFTI,
                 legacy_vtk_format=False, bbox_valid_check=True,
                 ref_pial=None):
    """
    Save the stateful surface to any format (vtk/vtp/obj/stl/ply/gii/pial)

    Parameters
    ----------
    fname : str
        Absolute path of the file.
    sfs : StatefulSurface
        The surface to save (must have been loaded properly)
    to_space : Enum (dipy.io.stateful_surface.Space)
        Space to which the surface will be transformed before saving
    to_origin : Enum (dipy.io.stateful_surface.Origin)
        Origin to which the surface will be transformed before saving
            NIFTI standard, default (center of the voxel)
            TRACKVIS standard (corner of the voxel)
    legacy_vtk_format : bool
        Whether to save the file in legacy VTK format or not.
    check_bbox_valid : bool
        Verification for negative voxel coordinates or values above the
        volume dimensions. Default is True, to enforce valid file.
    ref_pial : str
        Reference pial file to save the surface in pial format.
        If not provided, the metadata of the input surface is used.

    Raises
    ------
    ValueError
        If the bounding box is not valid in voxel space.
    """
    old_space = deepcopy(sfs.space)
    old_origin = deepcopy(sfs.origin)
    if bbox_valid_check and not sfs.is_bbox_in_vox_valid():
        raise ValueError(
            "Bounding box is not valid in voxel space, cannot "
            "save a valid file if some coordinates are invalid.\n"
            "Please set bbox_valid_check to False and verify the "
            "bounding box of the surface.")

    _, ext = os.path.splitext(fname)

    if ext in [".vtk", ".vtp", ".obj", ".stl", ".ply"]:
        sfs.to_space(to_space)
        sfs.to_origin(to_origin)
        if sfs.data_per_point is not None:
            # Check if rgb, colors, colors, etc. are available
            color_array_name = None
            for key in ["rgb", "colors", "colors", "color"]:
                if key in sfs.data_per_point:
                    color_array_name = key
                    break
                if key.upper() in sfs.data_per_point:
                    color_array_name = key.upper()
                    break
        save_polydata(sfs.get_polydata(), fname, legacy_vtk_format=legacy_vtk_format,
                      color_array_name=color_array_name)
    elif ext in [".gii", ".gii.gz"]:
        # TODO: check if this is correct
        sfs.to_space(to_space)
        sfs.to_origin(to_origin)
        vertices, faces = sfs.get_vertices_faces()
        surf_img = nib.gifti.GiftiImage()
        surf_img.add_gifti_data_array(ns.numpy_to_vtk(vertices, deep=True))
        surf_img.add_gifti_data_array(ns.numpy_to_vtk(faces, deep=True))
        nib.save(surf_img, fname)
    elif ext == ".pial":
        if not hasattr(sfs, "metadata") and ref_pial is None:
            raise ValueError("Metadata is required to save a pial file.\n"
                             "Please provide the reference pial file.")

        if ref_pial is not None:
            _, ext = os.path.splitext(ref_pial)
            if ext != ".pial":
                raise ValueError(
                    "Reference pial file must have .pial extension.")
            metadata = load_pial(ref_pial, return_meta=True)[-1]
        else:
            metadata = sfs.metadata

        if to_space is not None or to_origin is not None:
            logging.warning(
                "to_space and to_origin are ignored when loading pial files.")
        sfs.to_space(Space.RASMM)
        sfs.to_origin(Origin.NIFTI)

        affine, dimensions = sfs.affine, sfs.dimensions
        center_volume = np.array(dimensions) / 2
        xform_translation = np.dot(
            affine[0:3, 0:3], center_volume) + affine[0:3, 3] + [0.5, 0.5, -0.5]

        vertices = deepcopy(sfs.vertices) - xform_translation
        print('final', vertices[0])
        save_pial(fname, vertices, sfs.faces, metadata)
    else:
        logging.error("Output extension is not one of the supported format.")

    sfs.to_space(old_space)
    sfs.to_origin(old_origin)


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
    not_valid_geometry = False
    try:
        data = nib.freesurfer.read_geometry(fname, read_metadata=return_meta)
    except ValueError:
        try:
            data = nib.freesurfer.read_geometry(fname, read_metadata=False)
            logging.warning(
                "No metadata found, please use a pial file with metadata.")
        except ValueError:
            raise ValueError(f"{fname} provided does not have geometry data.")

    return data


def save_pial(fname, vertices, faces, metadata):
    """Save pial file.

    Parameters
    ----------
    fname : str
        Absolute path of the file.
    vertices : ndarray
        Vertices.
    faces : ndarray
        Faces.
    metadata : dict
        Key-value pairs to encode at the end of the file.
    """
    nib.freesurfer.write_geometry(fname, vertices, faces, volume_info=metadata)


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
