"""

===========================
Read/Write streamline files
===========================

Overview
========

DIPY_ can read and write many different file formats. In this example
we give a short introduction on how to use it for loading or saving
streamlines.

Read :ref:`faq`

"""

import os

import nibabel as nib
import numpy as np
from dipy.data import get_fnames
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.utils import (create_nifti_header, create_tractogram_header,
                           get_reference_info, is_header_compatible)
from dipy.tracking.streamline import Streamlines

from dipy.data.fetcher import (fetch_file_formats,
                               get_file_formats)

fetch_file_formats()
bundles_filename, ref_anat_filename = get_file_formats()
for filename in bundles_filename:
    print(os.path.basename(filename))

# Loading a Nifti1Image using nibabel
reference_anatomy = nib.load(ref_anat_filename)

# Load tractogram will support 5 file formats, function like load_trk or
# load_tck will simply be restricted to one file format
cc_trk = load_tractogram(bundles_filename[0], reference_anatomy)

# TRK files contain their own header (when writen properly), so they
# technically do not need a reference. (See how below)
# cc_trk = load_tractogram(bundles_filename[0], 'same')

laf_tck = load_tractogram(bundles_filename[1], reference_anatomy)
raf_vtk = load_tractogram(bundles_filename[3], reference_anatomy)

# These file contains invalid streamlines (negative values once in voxel space)
# This is not considered a valid tractography file, but it is possible to load
# it anyway
lpt_fib = load_tractogram(bundles_filename[2], reference_anatomy,
                          bbox_valid_check=False)
rpt_dpy = load_tractogram(bundles_filename[4], reference_anatomy, 
    bbox_valid_check=False)

"""
The function load_tractogram requires a reference, any of the following input
is considered valid:
- Nifti filename
- Trk filename
- nib.nifti1.Nifti1Image
- nib.streamlines.trk.TrkFile
- nib.nifti1.Nifti1Header
- Trk header (dict)

The reason why this parameters is required is to be sure all information
related to space attribute are always present.
"""

affine, dimensions, voxel_sizes, voxel_order = get_reference_info(
    reference_anatomy)
print(affine)
print(dimensions)
print(voxel_sizes)
print(voxel_order)

"""
If you have a TRK file that was generated using a particular anatomy,
to be considered valid all fields must correspond in the header.
It can be easily verified using this function, which also accept
the same variety of input as 
"""
print(is_header_compatible(reference_anatomy, bundles_filename[1]))

"""
If a TRK was generated with a valid header, but the reference NIFTI was lost
an header can be generated to then generate a fake NIFTI file.
"""
header = create_nifti_header(affine, dimension, voxel_size)
nib.save(nib.Nifti1Image(np.zeros(dimensions), affine, header), 'fake.nii.gz')
