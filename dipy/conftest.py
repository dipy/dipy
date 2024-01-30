"""pytest initialization."""
import importlib
import warnings

import numpy as np
from packaging.version import Version
import pytest


""" Set numpy print options to "legacy" for new versions of numpy
 If imported into a file, pytest will run this before any doctests.

References
----------
https://github.com/numpy/numpy/commit/710e0327687b9f7653e5ac02d222ba62c657a718
https://github.com/numpy/numpy/commit/734b907fc2f7af6e40ec989ca49ee6d87e21c495
https://github.com/nipy/nibabel/pull/556
"""
np.set_printoptions(legacy='1.13')

warnings.simplefilter(action="default", category=FutureWarning)
warnings.simplefilter("always", category=UserWarning)
# List of files that pytest should ignore
collect_ignore = ["testing/decorators.py", "bench*.py", "**/benchmarks/*"]


def pytest_collect_file(parent, file_path):
    if file_path.suffix in [".pyx", ".so"] and  \
       file_path.name.startswith("test"):
        return PyxFile.from_parent(parent, path=file_path)


class PyxFile(pytest.File):
    def collect(self):
        try:
            mod_name = self.path.stem.split('.')[0]
            base = self.parent.module.__name__
            mod = importlib.import_module(f'{base}.{mod_name}')
            for name in dir(mod):
                item = getattr(mod, name)
                if callable(item) and name.startswith("test_"):
                    yield PyxItem.from_parent(self, name=name, test_func=item,
                                              mod=mod)
        except ImportError:
            msg = (f"Import failed for {self.path}. Make sure you cython file "
                   "has been compiled.")
            raise PyxException(msg, self.path, 0)


class PyxItem(pytest.Item):
    def __init__(self, *, test_func, mod, **kwargs):
        super().__init__(**kwargs)
        self.mod = mod
        self.test_func = test_func

    def runtest(self):
        """Called to execute the test item."""
        self.test_func()

    def repr_failure(self, excinfo):
        """Called when self.runtest() raises an exception."""
        return excinfo.value.args[0]

    def reportinfo(self):
        return self.path, 0, f"test: {self.name}"


class PyxException(Exception):
    """Custom exception for error reporting."""
