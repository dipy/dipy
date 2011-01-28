# init for externals package
""" Calls to external packages """

# Test callable
from numpy.testing import Tester
test = Tester().test
del Tester
