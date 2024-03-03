import nibabel as nib


def load_gifti(fname):
    data = nib.load(fname)
    print(data.agg_data('NIFTI_INTENT_TRIANGLE').shape)
