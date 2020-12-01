import pytest
from distutils.version import LooseVersion
from numpy.testing import assert_equal
from dipy.utils.optpkg import optional_package

from dipy.nn.metrics import normalized_cross_correlation_loss
from dipy.nn.metrics import mean_squared_error

tf, have_tf, _ = optional_package('tensorflow')

if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_normalized_cross_correlation_loss():
    criterion = normalized_cross_correlation_loss()
    x1 = tf.random.uniform((1028, 32, 32, 3), minval=0, maxval=255)
    x2 = tf.random.uniform((1028, 32, 32, 3), minval=0, maxval=255)
    x3 = tf.fill((1028, 32, 32, 3), tf.random.uniform((), minval=0,
                                                      maxval=255))
    x4 = tf.zeros_like(x1)

    ncc_x1_x1 = -criterion(x1, x1).numpy()  # should be 1
    ncc_x1_x2 = -criterion(x1, x2).numpy()
    ncc_x2_x1 = -criterion(x2, x1).numpy()  # same as ncc_x1_x2
    ncc_x3_x1 = -criterion(x3, x1).numpy()  # close to 0
    ncc_x4_x1 = -criterion(x4, x1).numpy()  # 0
    ncc_x4_x4 = -criterion(x4, x4).numpy()  # 0

    assert_equal(abs(ncc_x1_x1 - 1) < 1e-5, True)
    assert_equal(ncc_x1_x2, ncc_x2_x1)
    assert_equal(abs(ncc_x3_x1) < 1e-5, True)
    assert_equal(ncc_x4_x1, 0)
    assert_equal(ncc_x4_x4, 0)


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_mean_squared_error():
    criterion = mean_squared_error()
    x1 = tf.random.uniform((8, 32, 32, 3), minval=0, maxval=255)
    x2 = tf.random.uniform((8, 32, 32, 3), minval=0, maxval=255)
    x3 = tf.zeros_like(x1)

    mse_x1_x1 = criterion(x1, x1).numpy()  # should be 0
    mse_x1_x2 = criterion(x1, x2).numpy()
    mse_x2_x1 = criterion(x2, x1).numpy()  # same as mse_x1_x2
    mse_x3_x1 = criterion(x3, x1).numpy()  # = tf.reduce_mean(tf.square(x1))
    mse_x3_x3 = criterion(x3, x3).numpy()  # 0

    assert_equal(mse_x1_x1, 0)
    assert_equal(mse_x1_x2, mse_x2_x1)
    assert_equal(mse_x3_x1, tf.reduce_mean(tf.square(x1)).numpy())
    assert_equal(mse_x3_x3, 0)


if __name__ == "__main__":
    test_normalized_cross_correlation_loss()
    test_mean_squared_error()
