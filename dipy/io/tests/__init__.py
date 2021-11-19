# init to allow relative imports in tests
# Test callable
from numpy.testing import Tester
test = Tester().test
del Tester
