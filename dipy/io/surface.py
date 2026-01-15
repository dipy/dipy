from copy import deepcopy
import gzip
import logging
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from dipy.io.stateful_surface import StatefulSurface
from dipy.io.utils import (
    Origin,
    Space,
    get_reference_info,
    split_filename_extension,
)
from dipy.io.vtk import (
    get_polydata_triangles,
    get_polydata_vertices,
    load_polydata,
    save_polydata,
)
from dipy.testing.decorators import warning_for_keywords
from dipy.utils.optpkg import optional_package

vtk, have_vtk, setup_module = optional_package(
    "vtk", min_version="9.0.0", max_version="9.1.0"
)

if have_vtk:
    import vtk
    import vtk.util.numpy_support as ns


def load_surface(
    fname,
    reference,
    *,
    to_space=Space.RASMM,
    to_origin=Origin.NIFTI,
    bbox_valid_check=True,
    from_space=None,
    from_origin=None,
    gifti_in_freesurfer=False,
):
    """Load the stateful surface from any format (vtk/vtp/obj/stl/ply/gii/pial)

    Parameters
    ----------
    filename : string or Path
        Filename with valid extension
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict), or 'same' if the input is a trk file.
        Reference that provides the spatial attribute.
        Typically a nifti-related object from the native diffusion used for
        surface generation
    to_space : Enum (dipy.io.utils.Space), optional
        Space to which the surface will be transformed after loading
    to_origin : Enum (dipy.io.utils.Origin), optional
        Origin to which the surface will be transformed after loading
            NIFTI standard, default (center of the voxel)
            TRACKVIS standard (corner of the voxel)
    bbox_valid_check : bool, optional
        Verification for negative voxel coordinates or values above the
        volume dimensions. Default is True, to enforce valid file.
    from_space : Enum (dipy.io.utils.Space), optional
        Space to which the surface was transformed before saving.
        Help for software compatibility. If None, assumes RASMM.
    from_origin : Enum (dipy.io.utils.Origin), optional
        Origin to which the surface was transformed before saving.
        Help for software compatibility. If None, assumes NIFTI.
    gifti_in_freesurfer : bool, optional
        Whether the gifti file is in freesurfer reference space.

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
    freesurfer_ext = [".gii", ".gii.gz", ".pial", ".nofix", ".orig", ".smoothwm", ".T1"]

    name, ext = split_filename_extension(fname)
    if ext.lower() not in freesurfer_ext + vtk_ext:
        logging.error("Input extension is not one of the supported format.")
        return False

    if to_space not in Space:
        logging.error(
            f"Space MUST be one of the {len(Space)} choices:"
            f" {list(Space.__members__.keys())}."
        )
        return False

    if to_origin not in Origin:
        logging.error(
            f"Origin MUST be one of the {len(Origin)} choices:"
            f" {list(Origin.__members__.keys())}."
        )
        return False

    if reference == "same":
        if ext in [".gii", ".gii.gz"]:
            reference = fname
        else:
            logging.error(
                'Reference must be provided, "same" is only available for GII file.'
            )
            return False

    timer = time.time()
    metadata = None
    data_per_vertex = None
    if ext in [".gii", ".gii.gz"]:
        data = load_gifti(fname)
        if gifti_in_freesurfer:
            data = (apply_freesurfer_transform(data[0], reference, inv=True), data[1])
        vertices = np.array(data[0])
        faces = np.array(data[1])
    elif ext in [".vtk", ".vtp", ".obj", ".stl", ".ply"]:
        data = load_polydata(fname)
        vertices = get_polydata_vertices(data)
        faces = get_polydata_triangles(data)
        vertex_data = data.GetPointData()
        scalar_names = [
            vertex_data.GetArrayName(i) for i in range(vertex_data.GetNumberOfArrays())
        ]
        if scalar_names:
            for name in scalar_names:
                scalar = data.GetPointData().GetScalars(name)
                if name in data_per_vertex:
                    logging.warning(
                        f"Scalar {name} already in data_per_vertex, overwriting"
                    )
                data_per_vertex[name] = ns.vtk_to_numpy(scalar)
    else:
        data = load_pial(fname, return_meta=True)
        data, metadata = data[0:2], data[2]

        data = (apply_freesurfer_transform(data[0], reference, inv=True), data[1])
        if from_space is not None or from_origin is not None:
            logging.warning(
                "from_space and from_origin are ignored when loading pial files."
            )
        from_space = Space.RASMM
        from_origin = Origin.NIFTI
        vertices = np.array(data[0])
        faces = np.array(data[1])

    from_space = Space.RASMM if from_space is None else from_space
    from_origin = Origin.NIFTI if from_origin is None else from_origin
    sfs = StatefulSurface(
        vertices,
        faces,
        reference,
        space=from_space,
        origin=from_origin,
        data_per_vertex=data_per_vertex,
    )
    if isinstance(metadata, dict):
        sfs.fs_metadata = metadata
    elif isinstance(metadata, nib.filebasedimages.FileBasedHeader):
        sfs.gii_header = metadata

    logging.debug(
        f"Load {fname} with {len(sfs)} vertices in {round(time.time() - timer, 3)}"
        f" seconds."
    )

    if bbox_valid_check and not sfs.is_bbox_in_vox_valid():
        raise ValueError(
            "Bounding box is not valid in voxel space, cannot "
            "load a valid file if some coordinates are invalid.\n"
            "Please set bbox_valid_check to False and be careful if processing "
            "the surface/vertices further."
        )

    sfs.to_space(to_space)
    sfs.to_origin(to_origin)

    return sfs


def save_surface(
    sfs,
    fname,
    *,
    to_space=Space.RASMM,
    to_origin=Origin.NIFTI,
    legacy_vtk_format=False,
    bbox_valid_check=True,
    ref_pial=None,
    ref_gii=None,
    gifti_in_freesurfer=False,
):
    """
    Save the stateful surface to any format (vtk/vtp/obj/stl/ply/gii/pial)

    Parameters
    ----------
    sfs : StatefulSurface
        The surface to save (must have been loaded properly)
    fname : str or Path
        Absolute path of the file.
    to_space : Enum (dipy.io.stateful_surface.Space), optional
        Space to which the surface will be transformed before saving
    to_origin : Enum (dipy.io.stateful_surface.Origin), optional
        Origin to which the surface will be transformed before saving
            NIFTI standard, default (center of the voxel)
            TRACKVIS standard (corner of the voxel)
    legacy_vtk_format : bool, optional
        Whether to save the file in legacy VTK format or not.
    bbox_valid_check : bool, optional
        Verification for negative voxel coordinates or values above the
        volume dimensions. Default is True, to enforce valid file.
    ref_pial : str or Path, optional
        Reference pial file to save the surface in pial format.
        If not provided, the metadata of the input surface is used (if available).
    ref_gii : str or Path, optional
        Reference gii file to save the surface in gii format.
        If not provided, the header of the input surface is used (if available).
    gifti_in_freesurfer : bool, optional
        Whether the gifti file must be saved in freesurfer reference space.

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
            "bounding box of the surface."
        )

    _, ext = split_filename_extension(fname)
    ext = ext.lower()

    if ext in [".vtk", ".vtp", ".obj", ".stl", ".ply"]:
        sfs.to_space(to_space)
        sfs.to_origin(to_origin)

        if sfs.data_per_vertex is not None:
            # Check if rgb, colors, colors, etc. are available
            color_array_name = None
            for key in ["rgb", "colors", "color"]:
                if key in sfs.data_per_vertex:
                    color_array_name = key
                    break
                if key.upper() in sfs.data_per_vertex:
                    color_array_name = key.upper()
                    break

        polydata = sfs.get_polydata()
        if color_array_name is not None:
            color_array = sfs.data_per_vertex[color_array_name]
            if len(color_array) != polydata.GetNumberOfPoints():
                raise ValueError("Array length does not match number of vertices.")
            vtk_array = ns.numpy_to_vtk(
                np.array(color_array), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
            )
            vtk_array.SetName("RGB")

            if "normal" in color_array_name.lower():
                polydata.GetPointData().SetNormals(vtk_array)
            else:
                try:
                    polydata.GetPointData().SetScalars(vtk_array)
                except ValueError:
                    polydata.GetPointData().AddArray(vtk_array)

        save_polydata(
            polydata, fname, legacy_vtk_format=legacy_vtk_format, color_array_name="RGB"
        )
    elif ext in [".gii", ".gii.gz"]:
        if not hasattr(sfs, "gii_header") and ref_gii is None:
            raise ValueError(
                "Metadata is required to save a gii file.\n"
                "Please provide the reference gii file."
            )

        if ref_gii is not None:
            _, ext = split_filename_extension(ref_gii)
            ext = ext.lower()
            if ext not in [".gii", ".gii.gz"]:
                raise ValueError("Reference gii file must have .gii extension.")
            _, metadata = load_gifti(ref_gii, return_header=True)[-1]
        else:
            metadata = sfs.gii_header

        sfs.to_space(to_space)
        sfs.to_origin(to_origin)
        if gifti_in_freesurfer:
            sfs.vertices = apply_freesurfer_transform(sfs.vertices, sfs, inv=False)
        save_gifti(fname, sfs.vertices, sfs.faces, header=metadata)

    elif ext == ".pial":
        if not hasattr(sfs, "fs_metadata") and ref_pial is None:
            raise ValueError(
                "Metadata is required to save a pial file.\n"
                "Please provide the reference pial file."
            )

        if ref_pial is not None:
            ext = Path(ref_pial).suffix
            if ext != ".pial":
                raise ValueError("Reference pial file must have .pial extension.")
            metadata = load_pial(ref_pial, return_meta=True)[-1]
        else:
            metadata = sfs.fs_metadata

        if to_space is not None or to_origin is not None:
            logging.warning(
                "to_space and to_origin are ignored when loading pial files."
            )
        sfs.to_space(Space.RASMM)
        sfs.to_origin(Origin.NIFTI)

        vertices = apply_freesurfer_transform(sfs.vertices, sfs, inv=False)

        save_pial(fname, vertices, sfs.faces, metadata=metadata)
    else:
        logging.error("Output extension is not one of the supported format.")

    sfs.to_space(old_space)
    sfs.to_origin(old_origin)


@warning_for_keywords()
def load_pial(fname, *, return_meta=False):
    """Load pial file.

    Parameters
    ----------
    fname : str or Path
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
        data = nib.freesurfer.read_geometry(fname, read_metadata=return_meta)
    except ValueError:
        try:
            data = nib.freesurfer.read_geometry(fname, read_metadata=False)
            logging.warning("No metadata found, please use a pial file with metadata.")
        except ValueError:
            raise ValueError(f"{fname} provided does not have geometry data.") from None

    return data


def save_pial(fname, vertices, faces, *, metadata=None):
    """Save pial file.

    Parameters
    ----------
    fname : str or Path
        Absolute path of the file.
    vertices : ndarray
        Vertices.
    faces : ndarray
        Faces.
    metadata : dict, optional
        Key-value pairs to encode at the end of the file.
    """
    nib.freesurfer.write_geometry(fname, vertices, faces, volume_info=metadata)


def load_gifti(fname, *, return_header=False):
    """Load gifti file.

    Parameters
    ----------
    fname : str or Path
        Absolute path of the file.

    return_header : bool, optional
        Whether to read the header of the file or not, by default False.
        If True, returns a tuple with vertices, faces and header.

    Returns
    -------
    tuple
        (vertices, faces)
    """

    def reader(fname):
        _, ext = split_filename_extension(fname)
        ext = ext.lower()
        if ext == ".gii.gz":
            with gzip.GzipFile(fname) as gz:
                img = nib.GiftiImage.from_bytes(gz.read())
        else:
            img = nib.load(fname)
        return img

    if return_header:
        gifti_img = reader(fname)
        return gifti_img.agg_data(("pointset", "triangle")), gifti_img.header
    else:
        return reader(fname).agg_data(("pointset", "triangle"))


def save_gifti(fname, vertices, faces, *, header=None):
    """Save gifti file.
    https://netneurolab.github.io/neuromaps/_modules/neuromaps/images.html

    Parameters
    ----------
    fname : str or Path
        Absolute path of the file.
    vertices : ndarray
        Vertices.
    faces : ndarray
        Faces.
    header : nib.filebasedimages.FileBasedHeader
        Valid header for the gifti file, typically loaded from a reference GII
    """
    vert = nib.gifti.GiftiDataArray(
        vertices,
        "NIFTI_INTENT_POINTSET",
        "NIFTI_TYPE_FLOAT32",
        coordsys=nib.gifti.GiftiCoordSystem(3, 3),
    )
    tri = nib.gifti.GiftiDataArray(faces, "NIFTI_INTENT_TRIANGLE", "NIFTI_TYPE_INT32")
    img = nib.GiftiImage(darrays=[vert, tri])
    nib.save(img, fname)


def apply_freesurfer_transform(vertices, reference, *, inv=False):
    """
    Apply the freesurfer transform to the vertices to bring them to RASMM space
    or to bring them back to the original space.

    Parameters
    ----------
    vertices : ndarray
        Vertices to transform.
    reference : str or Path
        Reference file to get the transform from.
    inv : bool, optional
        True if loading the surface, False if saving the surface.
    """

    affine, dimensions, _, _ = get_reference_info(reference)
    center_volume = np.array(dimensions) / 2
    vertices_copy = vertices.copy()
    if inv:
        xform_translation = np.dot(affine[0:3, 0:3], center_volume) + affine[0:3, 3]
    else:
        xform_translation = -(np.dot(affine[0:3, 0:3], center_volume) + affine[0:3, 3])

    return vertices_copy + xform_translation
