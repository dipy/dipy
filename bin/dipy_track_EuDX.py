#! /usr/bin/env python

from __future__ import division, print_function
from dipy.fixes import argparse as arg
from dipy.workflows.tracking import track_EuDX


parser = arg.ArgumentParser()
parser.add_argument('ref', action='store',
                    help='Path of the reference file.')
parser.add_argument('peaks', action='store', metavar='peaks',
                    help='Path of the peaks values file')
parser.add_argument('peaks_index', action='store', metavar='bvecs',
                    help='Path of the peaks index file.')
parser.add_argument('--out_dir', type=str, default='',
                    help='Path to output the resulting tractogram')

if __name__ == "__main__":
    params = parser.parse_args()
    track_EuDX(params.ref, params.peaks, params.peaks_index,
                     params.out_dir)




