import numpy as np
from dipy.tracking.vox2track import track_counts
from dipy.tracking.utils import density_map
import nibabel as nib
from nibabel.trackvis import write, empty_header

grid = np.mgrid[1.1:1.8:3j,1.1:1.8:3j,.5:5]
grid = np.rollaxis(grid, 0, 4)

streamlines = []

for ii in grid:
    for jj in ii:
        streamlines.append(jj)

#Treat these streamlines as if they are in trackvis format and generate counts
counts_trackvis = density_map(streamlines, (4,4,5), (1,1,1))

#Treat these streamlines as if they are in nifti format and generate counts
counts_nifti = track_counts(streamlines, (4,4,5), (1,1,1), 
                            return_elements=False)

print("saving trk files and track_count volumes")
aff = np.eye(4)
aff[0, 0] = -1
img = nib.Nifti1Image(counts_trackvis.astype('int16'), aff)
nib.save(img, 'counts_trackvis.nii.gz')
img = nib.Nifti1Image(counts_nifti.astype('int16'), aff)
nib.save(img, 'counts_nifti.nii.gz')

hdr = empty_header()
hdr['voxel_size'] = (1,1,1)
hdr['voxel_order'] = 'las'
hdr['vox_to_ras'] = aff
hdr['dim'] = counts_nifti.shape

#Treat these streamlines like they are in trackvis format and save them
streamlines_trackvis = ((ii,None,None) for ii in streamlines)
write('slAsTrackvis.trk', streamlines_trackvis, hdr)

#Move these streamlines from nifti to trackvis format and save them
streamlines_nifti = ((ii+.5,None,None) for ii in streamlines)
write('slAsNifti.trk', streamlines_nifti, hdr)

"""
Trackvis:
A------------
| C |   |   |
----B--------
|   |   |   |
-------------
|   |   |   |
------------D

A = [0, 0]
B = [1, 1]
C = [.5, .5]
D = [3, 3]



Nifti:
A------------
| C |   |   |
----B--------
|   |   |   |
-------------
|   |   |   |
------------D

A = [-.5, -.5]
B = [.5, .5]
C = [0, 0]
D = [2.5, 2.5]
"""
