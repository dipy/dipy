import numpy as np
import math
import scipy.ndimage
import pytest
from distutils.version import LooseVersion
from numpy.testing import assert_equal, assert_array_almost_equal

from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package('tensorflow')

if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')

from dipy.nn.registration import FCN2d
from dipy.nn.metrics import normalized_cross_correlation_loss


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, x, static, batch_size=8, shuffle=False):
        self.x = x
        self.static = static
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.x)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        moving = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        moving = moving[..., np.newaxis]
        static = self.static[np.newaxis, ..., np.newaxis]
        static = np.repeat(static, repeats=moving.shape[0], axis=0)

        # Rescale to [0, 1].
        moving = moving.astype(np.float32)  # (N, 32, 32, 1)
        static = static.astype(np.float32)  # (N, 32, 32, 1)
        moving = moving / 255.0
        static = static / 255.0

        return {'moving': moving, 'static': static}, static

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.x)


def load_mnist_data(label=7):
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

    return x_train, x_val, static


model = FCN2d(input_shape=(32, 32, 1))
x_train, x_val, static = load_mnist_data()


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_fit_evaluate():
    epochs = 3
    lr = 0.004
    batch_size = 32
    loss = normalized_cross_correlation_loss()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    train_loader = DataLoader(x_train, static, batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(x_val, static, batch_size=batch_size,
                            shuffle=True)

    model.compile(loss=loss, optimizer=optimizer)
    hist = model.fit(train_loader, epochs=epochs,
                     validation_data=val_loader)
    cc = -hist.history['val_loss'][-1]
    val_loss = model.evaluate(val_loader)
    cc_evaluate = -val_loss

    assert_equal(cc > 0.8, True)
    assert_equal(abs(cc - cc_evaluate) < 0.001, True)


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_predict():
    batch_size = 32
    val_loader = DataLoader(x_val, static, batch_size=batch_size,
                            shuffle=True)
    moved = model.predict(val_loader)
    static_ = static[np.newaxis, ...]
    static_ = np.repeat(static_, repeats=x_val.shape[0], axis=0)

    cc = -normalized_cross_correlation_loss()(static_, moved)
    assert_equal(cc > 0.8, True)


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_save_load_weights(tmpdir):
    path = tmpdir.mkdir('weights').join('fcn2d.h5')
    batch_size = 32
    val_loader = DataLoader(x_val, static, batch_size=batch_size,
                            shuffle=True)
    val_loss1 = model.evaluate(val_loader)
    model.save_weights(path)

    # define new model and load saved weights
    loss = normalized_cross_correlation_loss()
    model2 = FCN2d(input_shape=(32, 32, 1), loss=loss)
    model2.load_weights(path)
    val_loss2 = model2.evaluate(val_loader)

    assert_equal(abs(val_loss1) > 0.8, True)
    assert_equal(abs(val_loss1 - val_loss2) < 0.001, True)


if __name__ == "__main__":
    test_fit_evaluate()
    test_predict()
    test_save_load_weights()
