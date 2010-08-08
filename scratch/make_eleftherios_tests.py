import os
from os.path import join as pjoin
from glob import glob

import numpy as np

import dicom

import nibabel.dicom.dicomreaders as didr

from dipy.io.vectors import vector_norm

data_dir = os.path.expanduser(
    "~/data/20100114_195840/Series_012_CBU_DTI_64D_1A")
dcm_dir=sorted(glob(data_dir+"/*.dcm"))
voxels = [(68, 53, 27),
          (69, 63, 27),
          (17, 19, 27)]

acqs = len(dcm_dir)

half = (5,5,5)

datae = [[] for i in range(len(voxels))]

for dcm_file in dcm_dir:
    data_file = dicom.read_file(dcm_file)
    arr = didr.mosaic_to_nii(data_file).get_data()
    for p_ind, p in enumerate(voxels):
        datae[p_ind].append(arr[(p[0]-half[0]):(p[0]+half[0]+1),
                                (p[1]-half[1]):(p[1]+half[1]+1),
                                (p[2]-half[2]):(p[2]+half[2]+1)])

for (i,d) in enumerate(datae):
    newshape = [acqs]+list(datae[i][0].shape)
    datae[i] = np.concatenate(datae[i]).reshape(newshape)



#    data.append(d_data)
#    q = didr.get_q_vector(data_file)
#    b = vector_norm(q)
#    g = q / b
#    gs.append(g)
#    bs.append(b)
