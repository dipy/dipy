# init to make tests into a package
# Test callable
from numpy.testing import Tester
test = Tester().test
del Tester
