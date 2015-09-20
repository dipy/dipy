import tempfile
import os.path as op

import nose
import numpy.testing as npt
import dipy.data.fetcher as fetch


def test_check_md5():
    fd, fname = tempfile.mkstemp()
    stored_md5 = fetch._get_file_md5(fname)
    # If all is well, this shouldn't return anything:
    npt.assert_equal(fetch.check_md5(fname, stored_md5), None)
    # If None is provided as input, it should silently not check either:
    npt.assert_equal(fetch.check_md5(fname, None), None)
    # Otherwise, it will raise its exception class:
    npt.assert_raises(fetch.MD5Error, fetch.check_md5, fname, 'foo')


def test_make_fetcher():
    fd, fname = tempfile.mkstemp()
    tempdir =  tempfile.gettempdir()
    stored_md5 = fetch._get_file_md5(fname)
    baseurl = 'http://example.com'
    # Need to mock into urllib right here, so that it behaves like I hit
    # example.com looking for fname there, while getting it from my file-system
    # instead
    silly_fetcher = fetch._make_fetcher("silly_fetcher",
                                        tempdir, baseurl, [fname],
                                        ["silly_name"],
                                        md5_list=[stored_md5])

    silly_fetcher()
    assert op.isfile(op.join(tempdir, "silly_name"))
