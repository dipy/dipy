#! /usr/bin/env python
import nibabel as nib
import numpy as np
import argparse
import dipy.io.gradients as read
import textwrap

from dipy.reconst.shm import sf_to_sh
from dipy.core.sphere import Sphere

def sh_estimate(inFile, dirsInFile, outFile, rank=4, smoothness=0.0):
    in_nifti = nib.load(inFile)    
    refaff  = in_nifti.get_affine()
    data=in_nifti.get_data()

    vertices = np.loadtxt( dirsInFile )
    sphere = Sphere( xyz=vertices )

    odf_sh = sf_to_sh( data, sphere, int(rank), "mrtrix", smoothness )
           
    sh_out = nib.Nifti1Image(odf_sh.astype('float32'), refaff)
    nib.save(sh_out, outFile)

DESCRIPTION = 'Spherical harmonics (SH) estimation from any spherical function'

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('-i', action='store', dest='inFile',
                   metavar='FILE', required = True,
                   help='Input nifti file representing the spherical function on N vertices')
    p.add_argument('-s', action='store', dest='dirsInFile',
                   metavar='FILE', required = True,
                   help="""Sphere vertices in a text file (Nx3)
    x1 x2 x3
     ... 
    xN yN zN""")
    p.add_argument('-o', action='store', dest='outFile',
                   metavar='FILE', required = True,
                   help='Output nifti file')    
    p.add_argument('-r', action='store', dest='rank',
                   metavar='int', required = False, default = 8,
                   help='Maximum SH order of estimation (default 8)')    
    p.add_argument('-L', action='store', dest='smoothness',
                   metavar='float', required = False, default = 0,
                   help='Laplace-Beltrami regularization lambda (default 0)')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    
    inFile = args.inFile
    dirsInFile = args.dirsInFile
    outFile = args.outFile
    rank = args.rank
    smoothness = args.smoothness

    sh_estimate(inFile, dirsInFile, outFile, rank, smoothness)
    
if __name__ == "__main__":
    main()

