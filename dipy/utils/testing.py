from pytest import approx
import sys
import os
from numpy.core import around, number, float_, result_type, array
from numpy.core.numerictypes import issubdtype
from numpy.core.fromnumeric import any as npany
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
	assert actual == approx(desired)

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

def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    def compare(x, y):
        try:
            if npany(gisinf(x)) or npany( gisinf(y)):
                xinfid = gisinf(x)
                yinfid = gisinf(y)
                if not (xinfid == yinfid).all():
                    return False
                # if one item, x and y is +- inf
                if x.size == y.size == 1:
                    return x == y
                x = x[~xinfid]
                y = y[~yinfid]
        except (TypeError, NotImplementedError):
            pass


        dtype = result_type(y, 1.)
        y = array(y, dtype=dtype, copy=False, subok=True)
        z = abs(x - y)

        if not issubdtype(z.dtype, number):
            z = z.astype(float_)  # handle object arrays

        return z < 1.5 * 10.0**(-decimal)

    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
             header=('Arrays are not almost equal to %d decimals' % decimal),
             precision=decimal)

def assert_array_compare(comparison, x, y, err_msg='', verbose=True,
                         header='', precision=6, equal_nan=True,
                         equal_inf=True):
    x = array(x, copy=False, subok=True)
    y = array(y, copy=False, subok=True)

    def isnumber(x):
        return x.dtype.char in '?bhilqpBHILQPefdgFDG'

    def istime(x):
        return x.dtype.char in "Mm"

    def chk_same_position(x_id, y_id, hasval='nan'):
        """Handling nan/inf: check that x and y have the nan/inf at the same
        locations."""
        try:
            assert_array_equal(x_id, y_id)
        except AssertionError:
            msg = build_err_msg([x, y],
                                err_msg + '\nx and y %s location mismatch:'
                                % (hasval), verbose=verbose, header=header,
                                names=('x', 'y'), precision=precision)
            raise AssertionError(msg)

    try:
        cond = (x.shape == () or y.shape == ()) or x.shape == y.shape
        if not cond:
            msg = build_err_msg([x, y],
                                err_msg
                                + '\n(shapes %s, %s mismatch)' % (x.shape,
                                                                  y.shape),
                                verbose=verbose, header=header,
                                names=('x', 'y'), precision=precision)
            raise AssertionError(msg)

        if isnumber(x) and isnumber(y):
            has_nan = has_inf = False
            if equal_nan:
                x_isnan, y_isnan = isnan(x), isnan(y)
                # Validate that NaNs are in the same place
                has_nan = any(x_isnan) or any(y_isnan)
                if has_nan:
                    chk_same_position(x_isnan, y_isnan, hasval='nan')

            if equal_inf:
                x_isinf, y_isinf = isinf(x), isinf(y)
                # Validate that infinite values are in the same place
                has_inf = any(x_isinf) or any(y_isinf)
                if has_inf:
                    # Check +inf and -inf separately, since they are different
                    chk_same_position(x == +inf, y == +inf, hasval='+inf')
                    chk_same_position(x == -inf, y == -inf, hasval='-inf')

            if has_nan and has_inf:
                x = x[~(x_isnan | x_isinf)]
                y = y[~(y_isnan | y_isinf)]
            elif has_nan:
                x = x[~x_isnan]
                y = y[~y_isnan]
            elif has_inf:
                x = x[~x_isinf]
                y = y[~y_isinf]

            # Only do the comparison if actual values are left
            if x.size == 0:
                return

        elif istime(x) and istime(y):
            # If one is datetime64 and the other timedelta64 there is no point
            if equal_nan and x.dtype.type == y.dtype.type:
                x_isnat, y_isnat = isnat(x), isnat(y)

                if any(x_isnat) or any(y_isnat):
                    chk_same_position(x_isnat, y_isnat, hasval="NaT")

                if any(x_isnat) or any(y_isnat):
                    x = x[~x_isnat]
                    y = y[~y_isnat]

        val = comparison(x, y)

        if isinstance(val, bool):
            cond = val
            reduced = [0]
        else:
            reduced = val.ravel()
            cond = reduced.all()
            reduced = reduced.tolist()
        if not cond:
            match = 100-100.0*reduced.count(1)/len(reduced)
            msg = build_err_msg([x, y],
                                err_msg
                                + '\n(mismatch %s%%)' % (match,),
                                verbose=verbose, header=header,
                                names=('x', 'y'), precision=precision)
            raise AssertionError(msg)
    except ValueError:
        import traceback
        efmt = traceback.format_exc()
        header = 'error during assertion:\n\n%s\n\n%s' % (efmt, header)

        msg = build_err_msg([x, y], err_msg, verbose=verbose, header=header,
                            names=('x', 'y'), precision=precision)
        raise ValueError(msg)
