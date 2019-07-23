from copy import deepcopy
import logging
import os
import time

import nibabel as nib
from nibabel.streamlines import detect_format
from nibabel.streamlines.tractogram import Tractogram
import numpy as np

from dipy.io.stateful_tractogram import (create_tractogram_header,
                                         is_header_compatible,
                                         StatefulTractogram, Space)
from dipy.io.vtk import save_vtk_streamlines, load_vtk_streamlines
from dipy.io.dpy import Dpy


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
        TypeError('Output filename is not one of the supported format')

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError('Bounding box is not valid in voxel space, cannot ' +
                         'save a valid file if some coordinates are invalid')

    old_space = deepcopy(sft.get_current_space())
    old_shift = deepcopy(sft.get_current_shift())

    sft.to_rasmm()
    sft.to_center()

    timer = time.time()
    if extension in ['.trk', '.tck']:
        tractogram_type = detect_format(filename)
        header = create_tractogram_header(tractogram_type,
                                          *sft.get_space_attribute())
        new_tractogram = Tractogram(sft.get_streamlines(),
                                    affine_to_rasmm=np.eye(4))

        if extension == '.trk':
            new_tractogram.data_per_point = sft.get_data_per_point()
            new_tractogram.data_per_streamline = sft.get_data_per_streamline()

        fileobj = tractogram_type(new_tractogram, header=header)
        nib.streamlines.save(fileobj, filename)

    elif extension in ['.vtk', '.fib']:
        save_vtk_streamlines(sft.get_streamlines(), filename, binary=True)
    elif extension in ['.dpy']:
        dpy_obj = Dpy(filename, mode='w')
        dpy_obj.write_tracks(sft.get_streamlines())
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

    if extension == '.trk' and reference == 'same':
        reference = filename

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
