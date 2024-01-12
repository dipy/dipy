
"""This module is dedicated to the handling of tractograms."""
import os
import logging

import numpy as np
import trx.trx_file_memmap as tmm


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
                if group_key in all_groups_dtype and \
                    trx_1.groups[group_key].dtype !=  \
                        all_groups_dtype[group_key]:
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
