from copy import deepcopy
import logging
import os
import time

import nibabel as nib
from nibabel.streamlines import detect_format
from nibabel.streamlines.tractogram import Tractogram
import numpy as np

from dipy.io.stateful_tractogram import Space, StatefulTractogram
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

    Returns
    -------
    output : bool
        Did the saving work properly
    """

    _, extension = os.path.splitext(filename)
    if extension not in ['.trk', '.tck', '.vtk', '.fib', '.dpy']:
        raise TypeError('Output filename is not one of the supported format')

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError('Bounding box is not valid in voxel space, cannot ' +
                         'save a valid file if some coordinates are invalid')

    old_space = deepcopy(sft.space)
    old_shift = deepcopy(sft.shifted_origin)

    sft.to_rasmm()
    sft.to_center()

    timer = time.time()
    if extension in ['.trk', '.tck']:
        tractogram_type = detect_format(filename)
        header = create_tractogram_header(tractogram_type,
                                          *sft.space_attribute)
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

    if old_space == Space.VOX:
        sft.to_vox()
    elif old_space == Space.VOXMM:
        sft.to_voxmm()

    if old_shift:
        sft.to_corner()

    return True


def load_tractogram(filename, reference, to_space=Space.RASMM,
                    shifted_origin=False, bbox_valid_check=True,
                    trk_header_check=True):
    """ Load the stateful tractogram from any format (trk, tck, fib, dpy)

    Parameters
    ----------
    filename : string
        Filename with valid extension
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict), or 'same' if the input is a trk file.
        Reference that provides the spatial attribute.
        Typically a nifti-related object from the native diffusion used for
        streamlines generation
    space : string
        Space in which the streamlines will be transformed after loading
        (vox, voxmm or rasmm)
    shifted_origin : bool
        Information on the position of the origin,
        False is Trackvis standard, default (center of the voxel)
        True is NIFTI standard (corner of the voxel)

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
            logging.error('Reference must be provided, "same" is only ' +
                          'available for Trk file.')
            return False

    if trk_header_check and extension == '.trk':
        if not is_header_compatible(filename, reference):
            logging.error('Trk file header does not match the provided ' +
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
                             shifted_origin=shifted_origin,
                             data_per_point=data_per_point,
                             data_per_streamline=data_per_streamline)

    if to_space == Space.VOX:
        sft.to_vox()
    elif to_space == Space.VOXMM:
        sft.to_voxmm()

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError('Bounding box is not valid in voxel space, cannot ' +
                         'load a valid file if some coordinates are invalid')

    return sft


def load_trk(filename, reference, to_space=Space.RASMM,
             shifted_origin=False, bbox_valid_check=True,
             trk_header_check=True):
    _, extension = os.path.splitext(filename)
    if not extension == '.trk':
        raise ValueError('This function can only load trk file, for more'
                         ' generability use load_tractogram instead.')

    sft = load_tractogram(filename, reference,
                          to_space=Space.RASMM,
                          shifted_origin=shifted_origin,
                          bbox_valid_check=bbox_valid_check,
                          trk_header_check=trk_header_check)
    return sft


def load_tck(filename, reference, to_space=Space.RASMM,
             shifted_origin=False, bbox_valid_check=True,
             trk_header_check=True):
    _, extension = os.path.splitext(filename)
    if not extension == '.tck':
        raise ValueError('This function can only load tck file, for more'
                         ' generability use load_tractogram instead.')

    sft = load_tractogram(filename, reference,
                          to_space=Space.RASMM,
                          shifted_origin=shifted_origin,
                          bbox_valid_check=bbox_valid_check,
                          trk_header_check=trk_header_check)
    return sft


def load_vtk(filename, reference, to_space=Space.RASMM,
             shifted_origin=False, bbox_valid_check=True,
             trk_header_check=True):
    _, extension = os.path.splitext(filename)
    if not extension == '.vtk':
        raise ValueError('This function can only load vtk file, for more'
                         ' generability use load_tractogram instead.')

    sft = load_tractogram(filename, reference,
                          to_space=Space.RASMM,
                          shifted_origin=shifted_origin,
                          bbox_valid_check=bbox_valid_check,
                          trk_header_check=trk_header_check)
    return sft


def load_fib(filename, reference, to_space=Space.RASMM,
             shifted_origin=False, bbox_valid_check=True,
             trk_header_check=True):
    _, extension = os.path.splitext(filename)
    if not extension == '.fib':
        raise ValueError('This function can only load fib file, for more'
                         ' generability use load_tractogram instead.')

    sft = load_tractogram(filename, reference,
                          to_space=Space.RASMM,
                          shifted_origin=shifted_origin,
                          bbox_valid_check=bbox_valid_check,
                          trk_header_check=trk_header_check)
    return sft


def load_dpy(filename, reference, to_space=Space.RASMM,
             shifted_origin=False, bbox_valid_check=True,
             trk_header_check=True):
    _, extension = os.path.splitext(filename)
    if not extension == '.dpy':
        raise ValueError('This function can only load dpy file, for more'
                         ' generability use load_tractogram instead.')

    sft = load_tractogram(filename, reference,
                          to_space=Space.RASMM,
                          shifted_origin=shifted_origin,
                          bbox_valid_check=bbox_valid_check,
                          trk_header_check=trk_header_check)
    return sft


def save_trk(sft, filename, bbox_valid_check=True):
    _, extension = os.path.splitext(filename)
    if not extension == '.trk':
        raise ValueError('This function can only save trk file, for more'
                         ' generability use save_tractogram instead.')

    save_tractogram(sft, filename, bbox_valid_check=bbox_valid_check)


def save_tck(sft, filename, bbox_valid_check=True):
    _, extension = os.path.splitext(filename)
    if not extension == '.tck':
        raise ValueError('This function can only save tck file, for more'
                         ' generability use save_tractogram instead.')

    save_tractogram(sft, filename, bbox_valid_check=bbox_valid_check)


def save_vtk(sft, filename, bbox_valid_check=True):
    _, extension = os.path.splitext(filename)
    if not extension == '.vtk':
        raise ValueError('This function can only save vtk file, for more'
                         ' generability use save_tractogram instead.')

    save_tractogram(sft, filename, bbox_valid_check=bbox_valid_check)


def save_fib(sft, filename, bbox_valid_check=True):
    _, extension = os.path.splitext(filename)
    if not extension == '.fib':
        raise ValueError('This function can only save fib file, for more'
                         ' generability use save_tractogram instead.')

    save_tractogram(sft, filename, bbox_valid_check=bbox_valid_check)


def save_dpy(sft, filename, bbox_valid_check=True):
    _, extension = os.path.splitext(filename)
    if not extension == '.dpy':
        raise ValueError('This function can only save dpy file, for more'
                         ' generability use save_tractogram instead.')

    save_tractogram(sft, filename, bbox_valid_check=bbox_valid_check)


load_trk.__doc__ = load_tractogram.__doc__.replace(
    'from any format (trk, tck, fib, dpy)',
    'of the trk format')
load_tck.__doc__ = load_tractogram.__doc__.replace(
    'from any format (trk, tck, fib, dpy)',
    'of the tck format')
load_vtk.__doc__ = load_tractogram.__doc__.replace(
    'from any format (trk, tck, fib, dpy)',
    'of the vtk format')
load_fib.__doc__ = load_tractogram.__doc__.replace(
    'from any format (trk, tck, fib, dpy)',
    'of the fib format')
load_dpy.__doc__ = load_tractogram.__doc__.replace(
    'from any format (trk, tck, fib, dpy)',
    'of the dpy format')
save_trk.__doc__ = save_tractogram.__doc__.replace(
    'any format (trk, tck, vtk, fib, dpy)',
    'trk format')
save_tck.__doc__ = save_tractogram.__doc__.replace(
    'any format (trk, tck, vtk, fib, dpy)',
    'tck format')
save_vtk.__doc__ = save_tractogram.__doc__.replace(
    'any format (trk, tck, vtk, fib, dpy)',
    'vtk format')
save_fib.__doc__ = save_tractogram.__doc__.replace(
    'any format (trk, tck, vtk, fib, dpy)',
    'fib format')
save_dpy.__doc__ = save_tractogram.__doc__.replace(
    'any format (trk, tck, vtk, fib, dpy)',
    'dpy format')
