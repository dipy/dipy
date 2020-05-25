from dipy.utils.optpkg import optional_package
from distutils.version import LooseVersion

tf, have_tf, _ = optional_package("tensorflow")
layers, _, _ = optional_package("tensorflow.keras.layers")

if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')


def mean_squared_error():
    def mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
    return mse


def normalized_cross_correlation_loss():
    def ncc(y_true, y_pred):
        eps = tf.constant(1e-7, 'float32')
        y_true_mean = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
        y_pred_mean = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
        # shape (N, 1, 1, C)

        y_true_std = tf.math.reduce_std(y_true, axis=[1, 2], keepdims=True)
        y_pred_std = tf.math.reduce_std(y_pred, axis=[1, 2], keepdims=True)
        # shape (N, 1, 1, C)

        y_true_hat = (y_true - y_true_mean) / (y_true_std + eps)
        y_pred_hat = (y_pred - y_pred_mean) / (y_pred_std + eps)
        # shape (N, H, W, C)

        return -tf.reduce_mean(y_true_hat * y_pred_hat)  # shape ()
    return ncc
