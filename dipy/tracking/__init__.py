# Init for tracking module
""" Tracking objects """

from nibabel.streamlines import ArraySequence as Streamlines

# Test callable
from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
del Tester
