from pytest import approx
import sys
import os
import numpy as np


def assert_equal(value1, value2):
	try:
		assert np.all(value1 == value2)
	except AssertionError:
		raise AssertionError(str(value1) + ' != ' + str(value2))

def assert_not_equal(value1, value2):
	try:
		assert np.all(value1 != value2)
	except AssertionError:
		raise AssertionError(str(value1) + ' != ' + str(value2))

def assert_less(value1, value2):
	try:
		assert np.all(value1 < value2)
	except AssertionError:
		raise AssertionError(str(value1) + ' is not less than ' + str(value2))

def assert_less_equal(value1, value2):
	try:
		assert np.all(value1 <= value2)
	except AssertionError:
		raise AssertionError(str(value1) + ' is not less than ' + str(value2))

def assert_greater(value1, value2):
	try:
		assert np.all(value1 > value2)
	except AssertionError:
		raise AssertionError(str(value1) + ' is not greater than ' + str(value1))

def assert_greater_equal(value1, value2):
	try:
		assert np.all(value1 >= value2)
	except AssertionError:
		raise AssertionError(str(value1) + ' is not greater than ' + str(value1))

def assert_true(statement):
	try:
		assert statement
	except AssertionError:
		raise AssertionError('False is not true')

def assert_false(statement):
	try:
		assert not statement
	except AssertionError:
		raise AssertionError('True is not false')

def assert_raises(excClass, callableObj, *args, **kwargs):
	try:
		callableObj(*args, **kwargs)
	except excClass:
		return
	else:
		if hasattr(excClass, '__name__'): excName = excClass.__name__
		else: excName = str(excClass)
		raise AssertionError("{} not raised by {}".format(excName, callableObj))

def assert_almost_equal(actual, desired, decimal=7):
	if not abs(desired - actual) < 1.5 * 10**(-decimal):
		raise AssertionError("Values are not almost equal to {} decimals".format(decimal))

def assert_array_equal(actual, desired):
	assert np.all(actual == approx(desired))

def assert_array_almost_equal(actual, desired, decimal=7):
	assert np.all(actual == approx(desired, rel=10**(-1*decimal)))

def assert_(statement):
	assert statement

def assert_array_less(first, second):
	assert np.all(first < second)


'''Credits to the below functions goes to numpy.testing package'''


def assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
	def compare(x, y):
		return np.core.numeric.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

	actual, desired = np.asanyarray(actual), np.asanyarray(desired)
	header = 'Not equal to tolerance rtol=%g, atol=%g' % (rtol, atol)
	assert_almost_equal(actual, desired)

def measure():
	frame = sys._getframe(1)
	locs, globs = frame.f_locals, frame.f_globals

	code = compile(code_str,
	               'Test name: %s ' % label,
	                'exec')
	i = 0
	elapsed = jiffies()
	while i < times:
		i += 1
		exec(code, globs, locs)
	elapsed = jiffies() - elapsed
	return 0.01*elapsed


if sys.platform[:5] == 'linux':
	def jiffies(_proc_pid_stat='/proc/%s/stat' % (os.getpid()),
                _load_time=[]):
		"""
		Return number of jiffies elapsed.
		Return number of jiffies (1/100ths of a second) that this
		process has been scheduled in user mode. See man 5 proc.
		"""
		import time
		if not _load_time:
			_load_time.append(time.time())
		try:
			f = open(_proc_pid_stat, 'r')
			l = f.readline().split(' ')
			f.close()
			return int(l[13])
		except Exception:
			return int(100*(time.time()-_load_time[0]))
else:
    # os.getpid is not in all platforms available.
    # Using time is safe but inaccurate, especially when process
    # was suspended or sleeping.
	def jiffies(_load_time=[]):
		"""
		Return number of jiffies elapsed.
		Return number of jiffies (1/100ths of a second) that this
		process has been scheduled in user mode. See man 5 proc.
		"""
		import time
		if not _load_time:
			_load_time.append(time.time())
		return int(100*(time.time()-_load_time[0]))