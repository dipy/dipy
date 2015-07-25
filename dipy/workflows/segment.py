from dipy.utils.six import string_types
from glob import glob


def recognize_bundles_proxy(streamline_files, model_bundle_files,
                            model_streamline_files, out_dir=None):

    #print(streamline_files)
    #print(model_bundle_files)
    #print(model_streamline_files)
    #print(out_dir)

    if isinstance(streamline_files, string_types):
        sfiles = glob(streamline_files)
    else:
        raise ValueError('streamline files not a string')

    if isinstance(model_bundle_files, string_types):
        mbfiles = glob(model_bundle_files)

    for sf in sfiles:
        print(sf)
