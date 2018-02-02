from pytest import raises



def assert_equal(value1, value2):
	try:
		assert value1 == value2
	except AssertionError:
		raise AssertionError(str(value1) + ' != ' + str(value2))