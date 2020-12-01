import numpy as np
import scipy.ndimage
import pytest
from distutils.version import LooseVersion
from numpy.testing import assert_equal
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package('tensorflow')

if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')

    from dipy.nn.registration import UNet2d, RegistrationDataLoader
    from dipy.nn.metrics import normalized_cross_correlation_loss


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_unet2d(tmp_path):
    # Load data
    label = 7  # which digit images to train and test on
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Discard other digits.
    x_train = x_train[y_train == label].copy()  # (N, 28, 28)
    x_test = x_test[y_test == label].copy()

    # Resize to (32, 32)
    x_train = scipy.ndimage.zoom(x_train, (1, 32 / 28, 32 / 28))
    x_test = scipy.ndimage.zoom(x_test, (1, 32 / 28, 32 / 28))

    # Randomly choose static image from test set.
    idx = np.random.randint(x_test.shape[0])
    static = x_test[idx].copy()

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    static = static[np.newaxis, ..., np.newaxis]

    x_train = x_train.astype(np.float32)/255.0
    x_test = x_test.astype(np.float32)/255.0
    static = static.astype(np.float32)/255.0

    batch_size = 32
    epochs = 5
    lr = 0.007
    shuffle = True
    train_loader = RegistrationDataLoader(x_train, static,
                                          batch_size=batch_size,
                                          shuffle=shuffle)
    val_loader = RegistrationDataLoader(x_test, static, batch_size=batch_size,
                                        shuffle=shuffle)

    # Load model
    criterion = normalized_cross_correlation_loss()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    model = UNet2d(input_shape=(32, 32, 1), in_filters=4)
    model.compile(loss=criterion, optimizer=optimizer)

    # Test fit method
    hist = model.fit(train_loader, epochs=epochs, validation_data=val_loader)
    cc = -hist.history['val_loss'][-1]
    assert_equal(cc > 0.7, True)

    # Test evaluate method
    val_loss = model.evaluate(val_loader)
    cc_evaluate = -val_loss
    assert_equal(abs(cc - cc_evaluate) < 0.001, True)

    # Test predict method
    moved = model.predict(val_loader)
    static = np.repeat(static, repeats=moved.shape[0], axis=0)
    moved = tf.convert_to_tensor(moved)
    static = tf.convert_to_tensor(static)

    cc_predict = -criterion(static, moved).numpy()
    assert_equal(abs(cc_predict - cc_evaluate) < 0.001, True)

    # Test save_weights and load_weights
    path = tmp_path / "weights"
    path.mkdir()
    path = path / "unet2d.h5"

    val_loss1 = model.evaluate(val_loader)
    model.save_weights(str(path))

    # define new model and load saved weights
    model2 = UNet2d(input_shape=(32, 32, 1), loss=criterion, in_filters=4)
    model2.load_weights(str(path))
    val_loss2 = model2.evaluate(val_loader)

    assert_equal(abs(val_loss1) > 0.8, True)
    assert_equal(abs(val_loss1 - val_loss2) < 0.001, True)


if __name__ == "__main__":
    test_unet2d()
