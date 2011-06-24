#!/netopt/epd_py25/bin/python2.5

from numpy import array, prod
from nibabel import load
from nibabel.trackvis import aff_to_hdr, empty_header, write
from dipy.reconst.dti import Tensor
from dipy.io.bvectxt import read_bvec_file
from dipy.tracking.fact_tracking import seeds_from_mask, track_tensor
from IPython.kernel.client import MultiEngineClient

def foo():
    img = load('E8885S6I1.nii.gz')
    hdr = img.get_header()
    vox_size = hdr.get_zooms()[:3]
    data = img.get_data()
    data_mask = data[..., 0] > 250
    bvec, bval = read_bvec_file('E8885S6I1.bvec')
    roi_img = load('E8885S6I1_tracking_roi.nii.gz')
    roi_mask = roi_img.get_data()
    seeds = seeds_from_mask(roi_mask, [1,1,3])

    #making the tensors can be slow for large images
    tensor = Tensor(data, bval, bvec.T, data_mask, min_signal=1)
    start_step = array([0, 0, -1])

    #the tracking step can be slow for a large number of seeds
    tracks = track_tensor(tensor, seeds, start_step, vox_size, .25, 45)

    """
    mec = MultiEngineClient()
    mec.push({'tensor':tensor, 'start_step':start_step, 'vox_size':vox_size)
    mec.scatter('seeds', seeds)

    mec.execute('from dipy.tracking.fact_tracking import track_tensor')
    mec.exevute('tracks = track_tensor(tensor, seeds, start_step, vox_size, .25, 45)')

    tracks mec.gather('tracks')
    mec.kill('True')
    """

    tkvis_hdr = empty_header()
    tkvis_hdr['voxel_order']='LPI'
    tkvis_hdr['dim'] = tensor.shape
    tkvis_hdr['voxel_size'] = vox_size
    write('tracks.trk', tracks, tkvis_hdr)

