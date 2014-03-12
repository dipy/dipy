from os.path import join as pjoin, dirname
import nibabel as nib

test_piesno = nib.load(pjoin(dirname(__file__), 'test_piesno.nii.gz')).get_data()
