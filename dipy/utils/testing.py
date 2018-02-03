from pytest import raises



def assert_equal(value1, value2):
	try:
		assert value1 == value2
	except AssertionError:
		raise AssertionError(str(value1) + ' != ' + str(value2))


def assert_less(value1, value2):
	try:
		assert value1 < value2
	except AssertionError:
		raise AssertionError(str(value1) + ' is not less than ' + str(value2))

def assert_greater(value1, value2):
	try:
		assert value1 > value2
	except AssertionError:
		raise AssertionError(str(value1) + ' is not greater than ' + str(value1))

def assert_true(statement):
	try:
		assert statement
	except AssertionError:
		raise AssertionError('False is not true')

def assert_false(statement):
	try:
		assert statement
	except AssertionError:
		raise AssertionError('True is not false')

#TO_DO
assert_raises