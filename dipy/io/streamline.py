from copy import deepcopy
import logging
import os
import time

import nibabel as nib
from nibabel.streamlines import detect_format
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
import trx.trx_file_memmap as tmm

from dipy.io.dpy import Dpy
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.utils import Origin, Space, create_tractogram_header, is_header_compatible
from dipy.io.vtk import load_vtk_streamlines, save_vtk_streamlines
from dipy.testing.decorators import warning_for_keywords


@warning_for_keywords()
def save_tractogram(
    sft,
    filename,
    *,
    bbox_valid_check=True,
    to_space=Space.RASMM,
    to_origin=Origin.NIFTI,
):
    """Save the stateful tractogram in any format (trx/trk/tck/vtk/vtp/fib/dpy)

    Parameters
    ----------
    sft : StatefulTractogram
        The stateful tractogram to save
    filename : string
        Filename with valid extension
    bbox_valid_check : bool
        Verification for negative voxel coordinates or values above the
        volume dimensions. Default is True, to enforce valid file.
    to_space : Enum (dipy.io.utils.Space)
        Space to which the streamlines will be transformed before saving
    to_origin : Enum (dipy.io.utils.Origin)
        Origin to which the streamlines will be transformed before saving
            NIFTI standard, default (center of the voxel)
            TRACKVIS standard (corner of the voxel)
    """

    _, extension = os.path.splitext(filename)
    if extension not in [".trk", ".tck", ".trx", ".vtk", ".vtp", ".fib", ".dpy"]:
        raise TypeError("Output filename is not one of the supported format.")

    if to_space not in Space:
        raise ValueError("Space MUST be one of the 3 choices (Enum).")

    if to_origin not in Origin:
        raise ValueError("Origin MUST be one of the 2 choices (Enum).")

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError(
            "Bounding box is not valid in voxel space, cannot "
            "load a valid file if some coordinates are invalid.\n"
            "Please set bbox_valid_check to False and then use "
            "the function remove_invalid_streamlines to discard "
            "invalid streamlines."
        )

    old_space = deepcopy(sft.space)
    old_origin = deepcopy(sft.origin)

    timer = time.time()
    if extension in [".trk", ".tck", ".trx"]:
        to_origin = Origin.NIFTI
        to_space = Space.RASMM
        logging.warning(
            "to_space and to_origin are ignored when saving "
            ".trk or .tck or .trx files."
        )
    sft.to_space(to_space)
    sft.to_origin(to_origin)
    if extension in [".trk", ".tck"]:
        tractogram_type = detect_format(filename)
        header = create_tractogram_header(tractogram_type, *sft.space_attributes)
        new_tractogram = Tractogram(sft.streamlines, affine_to_rasmm=np.eye(4))

        if extension == ".trk":
            new_tractogram.data_per_point = sft.data_per_point
            new_tractogram.data_per_streamline = sft.data_per_streamline

        fileobj = tractogram_type(new_tractogram, header=header)
        nib.streamlines.save(fileobj, filename)

    elif extension in [".vtk", ".vtp", ".fib"]:
        binary = extension in [".vtk", ".fib"]
        save_vtk_streamlines(sft.streamlines, filename, binary=binary, to_lps=False)
        logging.warning(
            "StatefulTractogram was previously saving  in LPSMM space.\n"
            "Now use to_space=Space.LPSMM to match the previous behavior."
        )
    elif extension in [".dpy"]:
        dpy_obj = Dpy(filename, mode="w")
        dpy_obj.write_tracks(sft.streamlines)
        dpy_obj.close()
    elif extension in [".trx"]:
        trx = tmm.TrxFile.from_sft(sft)
        tmm.save(trx, filename)
        trx.close()

    logging.debug(
        "Save %s with %s streamlines in %s seconds.",
        filename,
        len(sft),
        round(time.time() - timer, 3),
    )

    sft.to_space(old_space)
    sft.to_origin(old_origin)


@warning_for_keywords()
def load_tractogram(
    filename,
    reference,
    *,
    to_space=Space.RASMM,
    to_origin=Origin.NIFTI,
    bbox_valid_check=True,
    from_space=None,
    from_origin=None,
    trk_header_check=True,
):
    """Load the stateful tractogram from any format (trx/trk/tck/vtk/vtp/fib/dpy)

    Parameters
    ----------
    filename : string
        Filename with valid extension
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict), or 'same' if the input is a trk file.
        Reference that provides the spatial attribute.
        Typically a nifti-related object from the native diffusion used for
        streamlines generation
    to_space : Enum (dipy.io.utils.Space)
        Space to which the streamlines will be transformed after loading
    to_origin : Enum (dipy.io.utils.Origin)
        Origin to which the streamlines will be transformed after loading
            NIFTI standard, default (center of the voxel)
            TRACKVIS standard (corner of the voxel)
    bbox_valid_check : bool
        Verification for negative voxel coordinates or values above the
        volume dimensions. Default is True, to enforce valid file.
    from_space : Enum (dipy.io.utils.Space)
        Space to which the tractogram was transformed before saving.
        Help for software compatibility. If None, assumes RASMM.
    from_origin : Enum (dipy.io.utils.Origin)
        Origin to which the tractogram was transformed before saving.
        Help for software compatibility. If None, assumes NIFTI.
    gifti_in_freesurfer : bool
    trk_header_check : bool
        Verification that the reference has the same header as the spatial
        attributes as the input tractogram when a Trk is loaded

    Returns
    -------
    output : StatefulTractogram
        The tractogram to load (must have been saved properly)
    """
    _, extension = os.path.splitext(filename)
    if extension not in [".trk", ".tck", ".trx", ".vtk", ".vtp", ".fib", ".dpy"]:
        logging.error("Output filename is not one of the supported format.")
        return False

    if to_space not in Space:
        logging.error("Space MUST be one of the 3 choices (Enum).")
        return False

    if reference == "same":
        if extension in [".trk", ".trx"]:
            reference = filename
        else:
            logging.error(
                'Reference must be provided, "same" is only available for Trk file.'
            )
            return False

    if trk_header_check and extension == ".trk":
        if not is_header_compatible(filename, reference):
            logging.error("Trk file header does not match the provided reference.")
            return False

    timer = time.time()
    data_per_point = None
    data_per_streamline = None
    if extension in [".trk", ".tck", ".trx"] and (
        from_space is not None or from_origin is not None
    ):
        from_space = None
        from_origin = None
        logging.warning(
            "from_space and from_origin are ignored when loading "
            ".trk or .tck or .trx files."
        )

    if extension in [".trk", ".tck"]:
        tractogram_obj = nib.streamlines.load(filename).tractogram
        streamlines = tractogram_obj.streamlines
        if extension == ".trk":
            data_per_point = tractogram_obj.data_per_point
            data_per_streamline = tractogram_obj.data_per_streamline

    elif extension in [".vtk", ".vtp", ".fib"]:
        streamlines = load_vtk_streamlines(filename, to_lps=False)
        logging.warning(
            "StatefulTractogram was previously saving in LPSMM space.\n"
            "Use from_space=Space.LPSMM to load older files."
        )
    elif extension in [".dpy"]:
        dpy_obj = Dpy(filename, mode="r")
        streamlines = list(dpy_obj.read_tracks())
        dpy_obj.close()

    from_space = Space.RASMM if from_space is None else from_space
    from_origin = Origin.NIFTI if from_origin is None else from_origin

    if extension in [".trx"]:
        trx_obj = tmm.load(filename)
        sft = trx_obj.to_sft()
        trx_obj.close()
    else:
        sft = StatefulTractogram(
            streamlines,
            reference,
            from_space,
            origin=from_origin,
            data_per_point=data_per_point,
            data_per_streamline=data_per_streamline,
        )

    logging.debug(
        "Load %s with %s streamlines in %s seconds.",
        filename,
        len(sft),
        round(time.time() - timer, 3),
    )

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError(
            "Bounding box is not valid in voxel space, cannot "
            "load a valid file if some coordinates are invalid.\n"
            "Please set bbox_valid_check to False and then use "
            "the function remove_invalid_streamlines to discard "
            "invalid streamlines."
        )

    sft.to_space(to_space)
    sft.to_origin(to_origin)

    return sft


def load_generator(ttype):
    """Generate a loading function that performs a file extension
    check to restrict the user to a single file format.

    Parameters
    ----------
    ttype : string
        Extension of the file format that requires a loader

    Returns
    -------
    output : function
        Function (load_tractogram) that handle only one file format
    """

    @warning_for_keywords()
    def f_gen(
        filename,
        reference,
        *,
        to_space=Space.RASMM,
        to_origin=Origin.NIFTI,
        bbox_valid_check=True,
        trk_header_check=True,
        from_space=None,
        from_origin=None,
    ):
        _, extension = os.path.splitext(filename)
        if not extension == ttype:
            msg = f"This function can only load {ttype} files, "
            msg += "for a more general purpose, use load_tractogram instead."
            raise ValueError(msg)

        sft = load_tractogram(
            filename,
            reference,
            to_space=to_space,
            to_origin=to_origin,
            bbox_valid_check=bbox_valid_check,
            trk_header_check=trk_header_check,
            from_space=from_space,
            from_origin=from_origin,
        )
        return sft

    f_gen.__doc__ = load_tractogram.__doc__.replace(
        "from any format (trk/tck/vtk/vtp/fib/dpy)", f"of the {ttype} format"
    )
    return f_gen


def save_generator(ttype):
    """Generate a saving function that performs a file extension
    check to restrict the user to a single file format.

    Parameters
    ----------
    ttype : string
        Extension of the file format that requires a saver

    Returns
    -------
    output : function
        Function (save_tractogram) that handle only one file format
    """

    def f_gen(sft, filename, bbox_valid_check=True):
        _, extension = os.path.splitext(filename)
        if not extension == ttype:
            msg = f"This function can only save {ttype} file, "
            msg += "for more general cases, use save_tractogram instead."
            raise ValueError(msg)
        save_tractogram(sft, filename, bbox_valid_check=bbox_valid_check)

    f_gen.__doc__ = save_tractogram.__doc__.replace(
        "in any format (trk/tck/vtk/vtp/fib/dpy)", f"of the {ttype} format"
    )
    return f_gen


load_trk = load_generator(".trk")
load_tck = load_generator(".tck")
load_trx = load_generator(".trx")
load_vtk = load_generator(".vtk")
load_vtp = load_generator(".vtp")
load_fib = load_generator(".fib")
load_dpy = load_generator(".dpy")
save_trk = save_generator(".trk")
save_tck = save_generator(".tck")
save_trx = save_generator(".trx")
save_vtk = save_generator(".vtk")
save_vtp = save_generator(".vtp")
save_fib = save_generator(".fib")
save_dpy = save_generator(".dpy")
