import os

from glob import glob

import numpy as np
import nibabel as nib
import tractconverter

from dipy.tracking.utils import density_map

def tract_density(tract_files, ref_files, up_factor, out_dir):
    for tract_file, ref_file in zip(glob(tract_files), glob(ref_files)):
        ref = nib.load(ref_file)
        ref_head = ref.get_header()
        pos_factor = ref_head['pixdim'][1:4] / up_factor
        data_shape = np.array(ref.shape) * up_factor
        data_shape = tuple(data_shape.astype('int32'))

        tract_format = tractconverter.detect_format(tract_file)
        tract = tract_format(tract_file, anatFile=ref_file)

        voxel_dim = ref_head['pixdim'][1:4]
        streamlines = [i for i in tract]

        affine = np.eye(4)
        #affine[:3, :3] *= np.asarray(voxel_dim)

        # Need to adjust the affine to take upsampling into account
        affine[0, 0] /= up_factor
        affine[1, 1] /= up_factor
        affine[2, 2] /= up_factor
        tdi_map = density_map(streamlines, data_shape, affine=affine)
        affine[:3, :] = ref.get_affine()[:3, :]

        map_img = nib.Nifti1Image(tdi_map.astype(np.float32), affine)

        if len(tdi_map.shape) > 3:
            pos_factor += [1]

        if out_dir == '':
            out_dir_path = os.path.dirname(ref_file)
        elif not os.path.isabs(out_dir):
            out_dir_path = os.path.join(os.path.dirname(ref_file), out_dir)
            if not os.path.exists(out_dir_path):
                os.makedirs(out_dir_path)
        else:
            out_dir_path = out_dir

        map_img.get_header().set_zooms(pos_factor)
        map_img.get_header().set_qform(ref_head.get_qform())
        map_img.get_header().set_sform(ref_head.get_sform())
        map_img.to_filename(os.path.join(out_dir_path,
                                         'tdi{0}x.nii.gz'.format(up_factor)))
