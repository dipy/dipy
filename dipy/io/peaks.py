from __future__ import division, print_function, absolute_import
import numpy as np
from dipy.direction.peaks import (PeaksAndMetrics,
                                  reshape_peaks_for_visualization)
from dipy.io.image import save_nifti


def load_peaks(fname):
    """ Load PeaksAndMetrics NPZ file
    """

    pam_dix = np.load(fname)

    pam = PeaksAndMetrics()
    pam.affine = pam_dix['affine']
    pam.peak_dirs = pam_dix['peak_dirs']
    pam.peak_values = pam_dix['peak_values']
    pam.peak_indices = pam_dix['peak_indices']
    pam.shm_coeff = pam_dix['shm_coeff']
    pam.sphere = pam_dix['sphere']
    pam.B = pam_dix['B']
    pam.invB = pam_dix['invB']
    pam.total_weight = pam_dix['total_weight']
    pam.ang_thr = pam_dix['ang_thr']
    pam.gfa = pam_dix['gfa']
    pam.qa = pam_dix['qa']
    pam.odf = pam_dix['odf']

    return pam


def save_peaks(fname, pam, compressed=True):
    """ Save NPZ file with all important attributes of object PeaksAndMetrics
    """

    if compressed:
        save_func = np.savez_compressed
    else:
        save_func = np.savez

    save_func(fname,
              affine=pam.affine,
              peak_dirs=pam.peak_dirs,
              peak_values=pam.peak_values,
              peak_indices=pam.peak_indices,
              shm_coeff=pam.shm_coeff,
              sphere=pam.sphere,
              B=pam.B,
              invB=pam.invB,
              total_weight=pam.total_weight,
              ang_thr=pam.ang_thr,
              gfa=pam.gfa,
              qa=pam.qa,
              odf=pam.odf)


def peaks_to_niftis(pam,
                    fname_shm,
                    fname_dirs,
                    fname_values,
                    fname_indices,
                    reshape_dirs=False):
        """ Save SH, directions, indices and values of peaks to Nifti.
        """

        save_nifti(fname_shm, pam.shm_coeff.astype(np.float32), pam.affine)

        if reshape_dirs:
            pam_dirs = reshape_peaks_for_visualization(pam)
        else:
            pam_dirs = pam.peak_dirs.astype(np.float32)

        save_nifti(fname_dirs, pam_dirs, pam.affine)

        save_nifti(fname_values, pam.peak_values.astype(np.float32),
                   pam.affine)

        save_nifti(fname_indices, pam.peak_indices, pam.affine)
