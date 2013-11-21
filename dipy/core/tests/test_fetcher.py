from os import path

import numpy.testing as npt
from nibabel.tmpdirs import TemporaryDirectory
from dipy.data import fetcher, SPHERE_FILES

def test_fetch_data():
    symmetric362 = SPHERE_FILES['symmetric362']
    with TemporaryDirectory() as tmpdir:
        md5 = fetcher._get_file_md5(symmetric362)
        bad_md5 = '8' * len(md5)

        newfile = path.join(tmpdir, "testfile.txt")
        # Test that the fetcher can get a file
        files = {"testfile.txt" : ("file://" + symmetric362, md5)}
        fetcher.fetch_data(files, tmpdir)
        npt.assert_(path.exists(newfile))

        # Test that the file is replaced when the md5 doesn't match
        with open(newfile, 'a') as f:
            f.write("some junk")
        fetcher.fetch_data(files, tmpdir)
        npt.assert_(path.exists(newfile))
        npt.assert_equal(fetcher._get_file_md5(newfile), md5)

        # Test that an error is raised when the md5 chechsum of the download
        # file does not match the expected value
        files = {"testfile.txt" : ("file://" + symmetric362, bad_md5)}
        npt.assert_raises(fetcher.FetcherError,
                          fetcher.fetch_data, files, tmpdir)

