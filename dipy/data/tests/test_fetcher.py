import tempfile
import os.path as op

try:
    from urllib import pathname2url
    from urlparse import urljoin
except ImportError:
    from urllib.request import pathname2url
    from urllib.parse import urljoin

import numpy.testing as npt
from nibabel.tmpdirs import TemporaryDirectory
import dipy.data.fetcher as fetcher
from dipy.data import SPHERE_FILES


def test_check_md5():
    fd, fname = tempfile.mkstemp()
    stored_md5 = fetcher._get_file_md5(fname)
    # If all is well, this shouldn't return anything:
    npt.assert_equal(fetcher.check_md5(fname, stored_md5), None)
    # If None is provided as input, it should silently not check either:
    npt.assert_equal(fetcher.check_md5(fname, None), None)
    # Otherwise, it will raise its exception class:
    npt.assert_raises(fetcher.FetcherError, fetcher.check_md5, fname, 'foo')


def test_make_fetcher():
    symmetric362 = SPHERE_FILES['symmetric362']
    with TemporaryDirectory() as tmpdir:
        stored_md5 = fetcher._get_file_md5(symmetric362)
        testfile_url = pathname2url(op.split(symmetric362)[0] + op.sep)
        testfile_url = urljoin("file:", testfile_url)
        print(testfile_url)
        print(symmetric362)
        sphere_fetcher = fetcher._make_fetcher("sphere_fetcher",
                                               tmpdir, testfile_url,
                                               [op.split(symmetric362)[-1]],
                                               ["sphere_name"],
                                               md5_list=[stored_md5])

        sphere_fetcher()
        assert op.isfile(op.join(tmpdir, "sphere_name"))
        npt.assert_equal(fetcher._get_file_md5(op.join(tmpdir, "sphere_name")),
                         stored_md5)

def test_fetch_data():
    symmetric362 = SPHERE_FILES['symmetric362']
    with TemporaryDirectory() as tmpdir:
        md5 = fetcher._get_file_md5(symmetric362)
        bad_md5 = '8' * len(md5)

        newfile = op.join(tmpdir, "testfile.txt")
        # Test that the fetcher can get a file
        testfile_url = pathname2url(symmetric362)
        testfile_url = urljoin("file:", testfile_url)
        files = {"testfile.txt": (testfile_url, md5)}
        fetcher.fetch_data(files, tmpdir)
        npt.assert_(op.exists(newfile))

        # Test that the file is replaced when the md5 doesn't match
        with open(newfile, 'a') as f:
            f.write("some junk")
        fetcher.fetch_data(files, tmpdir)
        npt.assert_(op.exists(newfile))
        npt.assert_equal(fetcher._get_file_md5(newfile), md5)

        # Test that an error is raised when the md5 checksum of the download
        # file does not match the expected value
        files = {"testfile.txt": (testfile_url, bad_md5)}
        npt.assert_raises(fetcher.FetcherError,
                          fetcher.fetch_data, files, tmpdir)
