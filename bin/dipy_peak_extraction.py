#! /usr/bin/env python

import nibabel as nib
import numpy as np
import argparse

from dipy.core.ndindex import ndindex
from dipy.reconst.odf import peak_directions
from dipy.core.sphere import Sphere


def peak_extraction(odfs_file, sphere_vertices_file, out_file, relative_peak_threshold=.25, min_separation_angle=45):
    in_nifti = nib.load(odfs_file)
    refaff = in_nifti.get_affine()
    odfs = in_nifti.get_data()

    vertices = np.loadtxt(sphere_vertices_file)
    sphere = Sphere(xyz=vertices)
    sphere.vertices = np.ascontiguousarray(sphere.vertices)
    sphere.edges = np.ascontiguousarray(sphere.edges)

    peaks = np.zeros(odfs.shape[:-1] + (15,))

    for index in ndindex(odfs.shape[:-1]):
        vox_peaks, _, _ = peak_directions(odfs[index], sphere, relative_peak_threshold, min_separation_angle)
        1/0
        peaks[index] = vox_peaks.ravel()[:15]

    peaks_img = nib.Nifti1Image(peaks.astype(np.float32), refaff)
    nib.save(peaks_img, out_file)


def buildArgsParser():
    description = 'Extract Peak Directions from Spherical function.'

    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument(action='store', dest='spherical_functions_file',
                   help='Input nifti file representing the orientation distribution function.')
    p.add_argument(action='store', dest='sphere_vertices_file',
                   help="""Sphere vertices in a text file (Nx3)
    x1 x2 x3
     ...
    xN yN zN""")
    p.add_argument(action='store', dest='out_file', help='Output nifti file with the peak directions.')
    # p.add_argument('-n', '--order', action='store', dest='rank',
    #                metavar='int', required=False, default=8,
    #                help='Maximum SH order of estimation (default 8)')
    # p.add_argument('-l', '--lambda', action='store', dest='smoothness',
    #                metavar='float', required=False, default=0.006,
    #                help='Laplace-Beltrami regularization (default 0.006)')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    spherical_functions_file = args.spherical_functions_file
    sphere_vertices_file = args.sphere_vertices_file
    out_file = args.out_file

    peak_extraction(spherical_functions_file, sphere_vertices_file, out_file,
                    relative_peak_threshold=.25, min_separation_angle=45)


if __name__ == "__main__":
    main()
