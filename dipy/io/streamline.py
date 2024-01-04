from copy import deepcopy
import logging
import os
import time

import nibabel as nib
from nibabel.streamlines import detect_format
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
import trx.trx_file_memmap as tmm

from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.vtk import save_vtk_streamlines, load_vtk_streamlines
from dipy.io.dpy import Dpy
from dipy.io.utils import (create_tractogram_header,
                           is_header_compatible)


def save_tractogram(sft, filename, bbox_valid_check=True):
    """ Save the stateful tractogram in any format (trk/tck/vtk/vtp/fib/dpy)

    Parameters
    ----------
    sft : StatefulTractogram
        The stateful tractogram to save
    filename : string
        Filename with valid extension
    bbox_valid_check : bool
        Verification for negative voxel coordinates or values above the
        volume dimensions. Default is True, to enforce valid file.

    Returns
    -------
    output : bool
        True if the saving operation was successful
    """

    _, extension = os.path.splitext(filename)
    if extension not in ['.trk', '.tck', '.trx', '.vtk', '.vtp', '.fib', '.dpy']:
        raise TypeError('Output filename is not one of the supported format.')

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError('Bounding box is not valid in voxel space, cannot '
                         'load a valid file if some coordinates are invalid.\n'
                         'Please set bbox_valid_check to False and then use '
                         'the function remove_invalid_streamlines to discard '
                         'invalid streamlines.')

    old_space = deepcopy(sft.space)
    old_origin = deepcopy(sft.origin)

    sft.to_rasmm()
    sft.to_center()

    timer = time.time()
    if extension in ['.trk', '.tck']:
        tractogram_type = detect_format(filename)
        header = create_tractogram_header(tractogram_type,
                                          *sft.space_attributes)
        new_tractogram = Tractogram(sft.streamlines,
                                    affine_to_rasmm=np.eye(4))

        if extension == '.trk':
            new_tractogram.data_per_point = sft.data_per_point
            new_tractogram.data_per_streamline = sft.data_per_streamline

        fileobj = tractogram_type(new_tractogram, header=header)
        nib.streamlines.save(fileobj, filename)

    elif extension in ['.vtk', '.vtp', '.fib']:
        binary = extension in ['.vtk', '.fib']
        save_vtk_streamlines(sft.streamlines, filename, binary=binary)
    elif extension in ['.dpy']:
        dpy_obj = Dpy(filename, mode='w')
        dpy_obj.write_tracks(sft.streamlines)
        dpy_obj.close()
    elif extension in ['.trx']:
        trx = tmm.TrxFile.from_sft(sft)
        tmm.save(trx, filename)
        trx.close()

    logging.debug('Save %s with %s streamlines in %s seconds.',
                  filename, len(sft), round(time.time() - timer, 3))

    sft.to_space(old_space)
    sft.to_origin(old_origin)

    return True


def load_tractogram(filename, reference, to_space=Space.RASMM,
                    to_origin=Origin.NIFTI, bbox_valid_check=True,
                    trk_header_check=True):
    """ Load the stateful tractogram from any format (trk/tck/vtk/vtp/fib/dpy)

    Parameters
    ----------
    filename : string
        Filename with valid extension
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict), or 'same' if the input is a trk file.
        Reference that provides the spatial attribute.
        Typically a nifti-related object from the native diffusion used for
        streamlines generation
    to_space : Enum (dipy.io.stateful_tractogram.Space)
        Space to which the streamlines will be transformed after loading
    to_origin : Enum (dipy.io.stateful_tractogram.Origin)
        Origin to which the streamlines will be transformed after loading
            NIFTI standard, default (center of the voxel)
            TRACKVIS standard (corner of the voxel)
    bbox_valid_check : bool
        Verification for negative voxel coordinates or values above the
        volume dimensions. Default is True, to enforce valid file.
    trk_header_check : bool
        Verification that the reference has the same header as the spatial
        attributes as the input tractogram when a Trk is loaded

    Returns
    -------
    output : StatefulTractogram
        The tractogram to load (must have been saved properly)
    """
    _, extension = os.path.splitext(filename)
    if extension not in ['.trk', '.tck', '.trx', '.vtk', '.vtp', '.fib', '.dpy']:
        logging.error('Output filename is not one of the supported format.')
        return False

    if to_space not in Space:
        logging.error('Space MUST be one of the 3 choices (Enum).')
        return False

    if reference == 'same':
        if extension in ['.trk', '.trx']:
            reference = filename
        else:
            logging.error('Reference must be provided, "same" is only '
                          'available for Trk file.')
            return False

    if trk_header_check and extension == '.trk':
        if not is_header_compatible(filename, reference):
            logging.error('Trk file header does not match the provided '
                          'reference.')
            return False

    timer = time.time()
    data_per_point = None
    data_per_streamline = None
    if extension in ['.trk', '.tck']:
        tractogram_obj = nib.streamlines.load(filename).tractogram
        streamlines = tractogram_obj.streamlines
        if extension == '.trk':
            data_per_point = tractogram_obj.data_per_point
            data_per_streamline = tractogram_obj.data_per_streamline

    elif extension in ['.vtk', '.vtp', '.fib']:
        streamlines = load_vtk_streamlines(filename)
    elif extension in ['.dpy']:
        dpy_obj = Dpy(filename, mode='r')
        streamlines = list(dpy_obj.read_tracks())
        dpy_obj.close()

    if extension in ['.trx']:
        trx_obj = tmm.load(filename)
        sft = trx_obj.to_sft()
        trx_obj.close()
    else:
        sft = StatefulTractogram(streamlines, reference, Space.RASMM,
                                 origin=Origin.NIFTI,
                                 data_per_point=data_per_point,
                                 data_per_streamline=data_per_streamline)

    logging.debug('Load %s with %s streamlines in %s seconds.',
                  filename, len(sft), round(time.time() - timer, 3))

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError('Bounding box is not valid in voxel space, cannot '
                         'load a valid file if some coordinates are invalid.\n'
                         'Please set bbox_valid_check to False and then use '
                         'the function remove_invalid_streamlines to discard '
                         'invalid streamlines.')

    sft.to_space(to_space)
    sft.to_origin(to_origin)

    return sft


def load_generator(ttype):
    """ Generate a loading function that performs a file extension
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
    def f_gen(filename, reference, to_space=Space.RASMM,
              to_origin=Origin.NIFTI, bbox_valid_check=True,
              trk_header_check=True):
        _, extension = os.path.splitext(filename)
        if not extension == ttype:
            msg = f"This function can only load {ttype} files, "
            msg += "for a more general purpose, use load_tractogram instead."
            raise ValueError(msg)

        sft = load_tractogram(filename, reference,
                              to_space=Space.RASMM,
                              to_origin=to_origin,
                              bbox_valid_check=bbox_valid_check,
                              trk_header_check=trk_header_check)
        return sft

    f_gen.__doc__ = load_tractogram.__doc__.replace(
        'from any format (trk/tck/vtk/vtp/fib/dpy)', f"of the {ttype} format")
    return f_gen


def save_generator(ttype):
    """ Generate a saving function that performs a file extension
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
        'in any format (trk/tck/vtk/vtp/fib/dpy)', f"of the {ttype} format")
    return f_gen


load_trk = load_generator('.trk')
load_tck = load_generator('.tck')
load_trx = load_generator('.trx')
load_vtk = load_generator('.vtk')
load_vtp = load_generator('.vtp')
load_fib = load_generator('.fib')
load_dpy = load_generator('.dpy')
save_trk = save_generator('.trk')
save_tck = save_generator('.tck')
save_trx = save_generator('.trx')
save_vtk = save_generator('.vtk')
save_vtp = save_generator('.vtp')
save_fib = save_generator('.fib')
save_dpy = save_generator('.dpy')


def concatenate_tractogram(tractogram_list, *, delete_dpv=False,
                           delete_dps=False, delete_groups=False,
                           check_space_attributes=True, preallocation=False):
    """Concatenate multiple tractograms into one.

    If the data_per_point or data_per_streamline is not the same for all
    tractograms, the data must be deleted first.

    Parameters
    ----------
    tractogram_list : List[StatefulTractogram or TrxFile]
        The stateful tractogram to concatenate
    delete_dpv: bool, optional
        Delete dpv keys that do not exist in all the provided TrxFiles
    delete_dps: bool, optional
        Delete dps keys that do not exist in all the provided TrxFile
    delete_groups: bool, optional
        Delete all the groups that currently exist in the TrxFiles
    check_space_attributes: bool, optional
        Verify that dimensions and size of data are similar between all the
        TrxFiles
    preallocation: bool, optional
        Preallocated TrxFile has already been generated and is the first
        element in trx_list (Note: delete_groups must be set to True as well)

    Returns
    -------
    new_trx: TrxFile
        TrxFile representing the concatenated data

    """
    trx_list = []
    for sft in tractogram_list:
        if not isinstance(sft, tmm.TrxFile):
            sft = tmm.TrxFile.from_sft(sft)
        elif len(sft.groups):
            delete_groups = True
        trx_list.append(sft)

    trx_list = [curr_trx for curr_trx in trx_list
                if curr_trx.header["NB_STREAMLINES"] > 0]

    if not trx_list:
        logging.warning("Inputs of concatenation were empty.")
        return tmm.TrxFile()

    if len(trx_list) == 1:
        if len(tractogram_list) > 1:
            logging.warning("Only 1 valid tractogram returned.")
        return trx_list[0]

    ref_trx = trx_list[0]
    all_dps = []
    all_dpv = []
    for curr_trx in trx_list:
        all_dps.extend(list(curr_trx.data_per_streamline.keys()))
        all_dpv.extend(list(curr_trx.data_per_vertex.keys()))
    all_dps, all_dpv = set(all_dps), set(all_dpv)

    if check_space_attributes:
        for curr_trx in trx_list[1:]:
            if not np.allclose(ref_trx.header["VOXEL_TO_RASMM"],
                               curr_trx.header["VOXEL_TO_RASMM"]) or \
                                   not np.array_equal(
                                       ref_trx.header["DIMENSIONS"],
                                       curr_trx.header["DIMENSIONS"]
            ):
                raise ValueError("Wrong space attributes.")

    if preallocation and not delete_groups:
        raise ValueError(
            "Groups are variables, cannot be handled with " "preallocation"
        )

    # Verifying the validity of fixed-size arrays, coherence between inputs
    for curr_trx in trx_list:
        for key in all_dpv:
            if key not in ref_trx.data_per_vertex.keys() \
                    or key not in curr_trx.data_per_vertex.keys():
                if not delete_dpv:
                    logging.debug(
                        "{} dpv key does not exist in all TrxFile.".format(key)
                    )
                    raise ValueError(
                        "TrxFile must be sharing identical dpv " "keys.")
            elif (
                ref_trx.data_per_vertex[key]._data.dtype
                != curr_trx.data_per_vertex[key]._data.dtype
            ):
                logging.debug(
                    "{} dpv key is not declared with the same dtype "
                    "in all TrxFile.".format(key)
                )
                raise ValueError("Shared dpv key, has different dtype.")

    for curr_trx in trx_list:
        for key in all_dps:
            if key not in ref_trx.data_per_streamline.keys() \
                    or key not in curr_trx.data_per_streamline.keys():
                if not delete_dps:
                    logging.debug(
                        "{} dps key does not exist in all " "TrxFile.".format(
                            key)
                    )
                    raise ValueError(
                        "TrxFile must be sharing identical dps " "keys.")
            elif (
                ref_trx.data_per_streamline[key].dtype
                != curr_trx.data_per_streamline[key].dtype
            ):
                logging.debug(
                    "{} dps key is not declared with the same dtype "
                    "in all TrxFile.".format(key)
                )
                raise ValueError("Shared dps key, has different dtype.")

    all_groups_len = {}
    all_groups_dtype = {}
    # Variable-size arrays do not have to exist in all TrxFile
    if not delete_groups:
        for trx_1 in trx_list:
            for group_key in trx_1.groups.keys():
                # Concatenating groups together
                if group_key in all_groups_len:
                    all_groups_len[group_key] += len(trx_1.groups[group_key])
                else:
                    all_groups_len[group_key] = len(trx_1.groups[group_key])
                if (group_key in all_groups_dtype and  \
                    trx_1.groups[group_key].dtype != all_groups_dtype[group_key]
                ):
                    raise ValueError("Shared group key, has different dtype.")
                else:
                    all_groups_dtype[group_key] = trx_1.groups[group_key].dtype

    # Once the checks are done, actually concatenate
    to_concat_list = trx_list[1:] if preallocation else trx_list
    if not preallocation:
        nb_vertices = 0
        nb_streamlines = 0
        for curr_trx in to_concat_list:
            curr_strs_len, curr_pts_len = curr_trx._get_real_len()
            nb_streamlines += curr_strs_len
            nb_vertices += curr_pts_len

        new_trx = tmm.TrxFile(
            nb_vertices=nb_vertices, nb_streamlines=nb_streamlines,
            init_as=ref_trx
        )
        if delete_dps:
            new_trx.data_per_streamline = {}
        if delete_dpv:
            new_trx.data_per_vertex = {}
        if delete_groups:
            new_trx.groups = {}

        tmp_dir = new_trx._uncompressed_folder_handle.name

        # When memory is allocated on the spot, groups and data_per_group can
        # be concatenated together
        for group_key in all_groups_len.keys():
            if not os.path.isdir(os.path.join(tmp_dir, "groups/")):
                os.mkdir(os.path.join(tmp_dir, "groups/"))
            dtype = all_groups_dtype[group_key]
            group_filename = os.path.join(
                tmp_dir, "groups/" "{}.{}".format(group_key, dtype.name)
            )
            group_len = all_groups_len[group_key]
            new_trx.groups[group_key] = tmm._create_memmap(
                group_filename, mode="w+", shape=(group_len,), dtype=dtype
            )
            if delete_groups:
                continue
            pos = 0
            count = 0
            for curr_trx in trx_list:
                curr_len = len(curr_trx.groups[group_key])
                new_trx.groups[group_key][pos: pos + curr_len] = \
                    curr_trx.groups[group_key] + count
                pos += curr_len
                count += curr_trx.header["NB_STREAMLINES"]

        strs_end, pts_end = 0, 0
    else:
        new_trx = ref_trx
        strs_end, pts_end = new_trx._get_real_len()

    for curr_trx in to_concat_list:
        # Copy the TrxFile fixed-size info (the right chunk)
        strs_end, pts_end = new_trx._copy_fixed_arrays_from(
            curr_trx, strs_start=strs_end, pts_start=pts_end
        )
    return new_trx

