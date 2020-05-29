from dipy.utils.optpkg import optional_package
from distutils.version import LooseVersion

tf, have_tf, _ = optional_package("tensorflow")
layers, _, _ = optional_package("tensorflow.keras.layers")

if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')


def mean_squared_error():
    def mse(y_true, y_pred):
        """Computes the mean squared error (MSE) loss.

        Parameters
        ----------
        y_true : tf.Tensor
            The static image to which the moving image is aligned.
        y_pred : tf.Tensor
            The moving image, the same shape as the static image.

        Returns
        -------
        loss : tf.Tensor, shape ()
            Mean squared error between the static and the moving images,
            averaged over the batch.

        """
        return tf.reduce_mean(tf.square(y_pred - y_true))
    return mse


def normalized_cross_correlation_loss():
    def ncc(y_true, y_pred):
        """Computes the normalized cross-correlation (NCC) loss.

        Parameters
        ----------
        y_true : tf.Tensor
            The static image to which the moving image is aligned.
        y_pred : tf.Tensor
            The moving image, the same shape as the static image.

        Returns
        -------
        loss : tf.Tensor, shape ()
            Normalized cross-correlation loss between the static and the
            moving images, averaged over the batch. Range is [-1.0, 1.0].
            The best value is -1 (perfect match) and the worst is 1.

        References
        ----------
        .. [1] `Wikipedia entry for the Cross-correlation
               <https://en.wikipedia.org/wiki/Cross-correlation>`_

        """
        eps = tf.constant(1e-7, 'float32')
        ndim = len(tf.keras.backend.int_shape(y_true))

        y_true_mean = tf.reduce_mean(y_true, axis=range(1, ndim-1),
                                     keepdims=True)
        y_pred_mean = tf.reduce_mean(y_pred, axis=range(1, ndim-1),
                                     keepdims=True)
        # shape (N, 1, 1, C)

        y_true_std = tf.math.reduce_std(y_true, axis=range(1, ndim-1),
                                        keepdims=True)
        y_pred_std = tf.math.reduce_std(y_pred, axis=range(1, ndim-1),
                                        keepdims=True)
        # shape (N, 1, 1, C)

        y_true_hat = (y_true - y_true_mean) / (y_true_std + eps)
        y_pred_hat = (y_pred - y_pred_mean) / (y_pred_std + eps)
        # shape (N, H, W, C)

        return -tf.reduce_mean(y_true_hat * y_pred_hat)  # shape ()
    return ncc
