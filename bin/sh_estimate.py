#! /usr/bin/env python
import nibabel as nib
import numpy as np
import argparse
import dipy.io.gradients as read

from reconst.shm_scil import normalize_data, shHARDIModel

DESCRIPTION = 'Spherical harmonics (SH) Estimation'

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('-dwi', action='store', dest='dwiFile',
                   metavar='FILE', required = True,
                   help='dwi file')
    p.add_argument('-r', action='store', dest='rank',
                   metavar='int', required = True, 
                   help='Maximum SH order of estimation')    
    p.add_argument('-b', action='store', dest='bFile',
                   metavar='FILE', required = True,
                   help='b-value file')
    p.add_argument('-g', action='store', dest='gradInFile',
                   metavar='FILE', required = True,
                   help='gradient directions input file')
    p.add_argument('-basisType', action='store', dest='basisType',
                   metavar='string', required = False, default = 2,
                   help='basis type: 0 for dipy, 1 for Descoteaux PhD thesis, 2 for mrtrix (default)')
    p.add_argument('-lambda', action='store', dest='smoothness',
                   metavar='float', required = False, default = 0.006,
                   help='Laplace-Beltrami regularization (default 0.006)')
    p.add_argument('-o', action='store', dest='outFile',
                   metavar='FILE', required = True,
                   help='output input file')
    
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    
    dwiFile = args.dwiFile
    bFile = args.bFile
    gradInFile = args.gradInFile
    rank = args.rank
    outFile = args.outFile
    basis_type = args.basisType
    smoothness = args.smoothness

    b_values , gradients = read.read_bvals_bvecs(bFile, gradInFile)
        
    dwi_nifti = nib.load(dwiFile)    
    refaff  = dwi_nifti.get_affine()
    dwi_data=dwi_nifti.get_data()

    dwi_data = normalize_data( dwi_data, b_values )

    hardi_model = shHARDIModel( b_values, gradients, int(rank), int(basis_type), smoothness)
    hardi_sh = hardi_model.fit( dwi_data )._shm_coef

    sh_out = nib.Nifti1Image(hardi_sh.astype('float32'), refaff)
    nib.save(sh_out, outFile)

    
if __name__ == "__main__":
    main()
    





