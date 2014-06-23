# Init for core dipy objects
""" Core objects """

# Test callable
from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
del Tester
