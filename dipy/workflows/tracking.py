#!/usr/bin/env python

from __future__ import division

import os
from glob import glob
import nibabel as nib
from dipy.tracking.eudx import EuDX
from dipy.data import get_sphere



def track_peaks_deterministic(ref, peaks, peaks_idx, out_dir):
    # Load Data volumes
    for ref_path, peaks_path, peaks_idx_path in zip(glob(ref),
                                                   glob(peaks),
                                                   glob(peaks_idx)):
        ref_img = nib.load(ref_path)
        ref_vol = ref_img.get_data()
        voxel_size = ref_img.get_header().get_zooms()[:3]

        peaks_idx = nib.load(peaks_idx_path).get_data()
        peaks_value = nib.load(peaks_path).get_data()

        print 'peaks: ', peaks_path
        print 'peaks idx: ', peaks_idx_path
        # Run Dipy EuDX Tracking
        tracks_iter = EuDX(peaks_value,
                           peaks_idx,
                           odf_vertices=get_sphere('symmetric362').vertices,
                           seeds=100000)

        # Save streamlines (tracking results)
        streamlines_trk = [(sl, None, None) for sl in tracks_iter]
        print 'after list comp'

        hdr = nib.trackvis.empty_header()
        hdr['voxel_size'] = voxel_size
        hdr['voxel_order'] = 'LAS'
        hdr['dim'] = ref_vol.shape
        hdr['n_count'] = len(streamlines_trk)

        if out_dir == '':
            out_dir_path = os.path.dirname(ref_path)
        elif not os.path.isabs(out_dir):
            out_dir_path = os.path.join(os.path.dirname(ref_path), out_dir)
            if not os.path.exists(out_dir_path):
                os.makedirs(out_dir_path)
        else:
            out_dir_path = out_dir

        nib.trackvis.write(os.path.join(out_dir_path, 'tractogram.trk'),
                           streamlines_trk,
                           hdr)
