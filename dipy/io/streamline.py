from copy import deepcopy
import logging
import os
import time

import nibabel as nib
from nibabel.streamlines import detect_format
from nibabel.streamlines.tractogram import Tractogram
import numpy as np

from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.vtk import save_vtk_streamlines, load_vtk_streamlines
from dipy.io.dpy import Dpy
from dipy.io.utils import (create_tractogram_header,
                           is_header_compatible)


def save_tractogram(sft, filename, bbox_valid_check=True):
    """ Save the stateful tractogram in any format (trk, tck, vtk, fib, dpy)

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
    if extension not in ['.trk', '.tck', '.vtk', '.fib', '.dpy']:
        raise TypeError('Output filename is not one of the supported format')

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError('Bounding box is not valid in voxel space, cannot '
                         'load a valid file if some coordinates are '
                         'invalid. Please use the function '
                         'remove_invalid_streamlines to discard invalid '
                         'streamlines or set bbox_valid_check to False')

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

    elif extension in ['.vtk', '.fib']:
        save_vtk_streamlines(sft.streamlines, filename, binary=True)
    elif extension in ['.dpy']:
        dpy_obj = Dpy(filename, mode='w')
        dpy_obj.write_tracks(sft.streamlines)
        dpy_obj.close()

    logging.debug('Save %s with %s streamlines in %s seconds',
                  filename, len(sft), round(time.time() - timer, 3))

    sft.to_space(old_space)
    sft.to_origin(old_origin)

    return True


def load_tractogram(filename, reference, to_space=Space.RASMM,
                    to_origin=Origin.NIFTI, bbox_valid_check=True,
                    trk_header_check=True):
    """ Load the stateful tractogram from any format (trk, tck, vtk, fib, dpy)

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
    if extension not in ['.trk', '.tck', '.vtk', '.fib', '.dpy']:
        logging.error('Output filename is not one of the supported format')
        return False

    if to_space not in Space:
        logging.error('Space MUST be one of the 3 choices (Enum)')
        return False

    if reference == 'same':
        if extension == '.trk':
            reference = filename
        else:
            logging.error('Reference must be provided, "same" is only '
                          'available for Trk file.')
            return False

    if trk_header_check and extension == '.trk':
        if not is_header_compatible(filename, reference):
            logging.error('Trk file header does not match the provided '
                          'reference')
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

    elif extension in ['.vtk', '.fib']:
        streamlines = load_vtk_streamlines(filename)
    elif extension in ['.dpy']:
        dpy_obj = Dpy(filename, mode='r')
        streamlines = list(dpy_obj.read_tracks())
        dpy_obj.close()
    logging.debug('Load %s with %s streamlines in %s seconds',
                  filename, len(streamlines), round(time.time() - timer, 3))

    sft = StatefulTractogram(streamlines, reference, Space.RASMM,
                             origin=Origin.NIFTI,
                             data_per_point=data_per_point,
                             data_per_streamline=data_per_streamline)

    sft.to_space(to_space)
    sft.to_origin(to_origin)

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError('Bounding box is not valid in voxel space, cannot '
                         'load a valid file if some coordinates are invalid.'
                         'Please set bbox_valid_check to False and then use'
                         'the function remove_invalid_streamlines to discard'
                         'invalid streamlines.')

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
            raise ValueError('This function can only load {} files, for a more'
                             ' general purpose, use load_tractogram instead.'.format(ttype))

        sft = load_tractogram(filename, reference,
                              to_space=Space.RASMM,
                              to_origin=to_origin,
                              bbox_valid_check=bbox_valid_check,
                              trk_header_check=trk_header_check)
        return sft

    f_gen.__doc__ = load_tractogram.__doc__.replace(
        'from any format (trk, tck, vtk, fib, dpy)',
        'of the {} format'.format(ttype))
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
            raise ValueError('This function can only save {} file, for more'
                             ' generability use save_tractogram instead.'.format(ttype))
        save_tractogram(sft, filename, bbox_valid_check=bbox_valid_check)

    f_gen.__doc__ = save_tractogram.__doc__.replace(
        'in any format (trk, tck, vtk, fib, dpy)',
        'of the {} format'.format(ttype))
    return f_gen


load_trk = load_generator('.trk')
load_tck = load_generator('.tck')
load_vtk = load_generator('.vtk')
load_fib = load_generator('.fib')
load_dpy = load_generator('.dpy')
save_trk = save_generator('.trk')
save_tck = save_generator('.tck')
save_vtk = save_generator('.vtk')
save_fib = save_generator('.fib')
save_dpy = save_generator('.dpy')
