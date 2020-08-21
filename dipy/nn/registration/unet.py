from ._utils import regular_grid_2d, grid_sample_2d
from dipy.utils.optpkg import optional_package
from distutils.version import LooseVersion

tf, have_tf, _ = optional_package("tensorflow")
layers, _, _ = optional_package("tensorflow.keras.layers")

if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')

__all__ = ["UNet2d"]


class UNet2d(object):
    def __init__(self, input_shape=(32, 32, 1), in_filters=64, multiplier=2,
                 optimizer='adam', loss=None, metrics=None, loss_weights=None):
        out_channels = 2

        moving = layers.Input(shape=input_shape, name='moving')
        static = layers.Input(shape=input_shape, name='static')

        x = layers.concatenate([moving, static], axis=-1)

        # encoder
        x1 = layers.Conv2D(in_filters, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x)   # 32 --> 32
        x1 = layers.Conv2D(in_filters, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x1)  # 32

        x2 = layers.MaxPool2D(pool_size=2)(x1)     # 32 --> 16
        x2 = layers.Conv2D(in_filters*multiplier, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x2)  # 16
        x2 = layers.Conv2D(in_filters*multiplier, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x2)  # 16

        x3 = layers.MaxPool2D(pool_size=2)(x2)     # 16 --> 8
        x3 = layers.Conv2D(in_filters*multiplier**2, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x3)  # 8
        x3 = layers.Conv2D(in_filters*multiplier**2, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x3)  # 8

        x4 = layers.MaxPool2D(pool_size=2)(x3)     # 8 --> 4
        x4 = layers.Conv2D(in_filters*multiplier**3, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x4)  # 4
        x4 = layers.Conv2D(in_filters*multiplier**3, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x4)  # 4

        x5 = layers.MaxPool2D(pool_size=2)(x4)     # 4 --> 2
        x5 = layers.Conv2D(in_filters*multiplier**4, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x5)  # 2
        x5 = layers.Conv2D(in_filters*multiplier**4, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x5)  # 2

        # decoder
        x6 = layers.UpSampling2D(size=2)(x5)       # 2 --> 4
        x6 = layers.Conv2D(in_filters*multiplier**3, kernel_size=2, strides=1,
                           padding='same',
                           activation=None)(x6)    # 4
        x6 = layers.concatenate([x6, x4], axis=-1)
        x6 = layers.Conv2D(in_filters*multiplier**3, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x6)  # 4
        x6 = layers.Conv2D(in_filters*multiplier**3, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x6)  # 4

        x6 = layers.UpSampling2D(size=2)(x6)       # 4 --> 8
        x6 = layers.Conv2D(in_filters*multiplier**2, kernel_size=2, strides=1,
                           padding='same',
                           activation=None)(x6)    # 8
        x6 = layers.concatenate([x6, x3], axis=-1)  # 8
        x6 = layers.Conv2D(in_filters*multiplier**2, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x6)  # 8
        x6 = layers.Conv2D(in_filters*multiplier**2, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x6)  # 8

        x6 = layers.UpSampling2D(size=2)(x6)       # 8 --> 16
        x6 = layers.Conv2D(in_filters*multiplier, kernel_size=2, strides=1,
                           padding='same',
                           activation=None)(x6)    # 16
        x6 = layers.concatenate([x6, x2], axis=-1)  # 16
        x6 = layers.Conv2D(in_filters*multiplier, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x6)  # 16
        x6 = layers.Conv2D(in_filters*multiplier, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x6)  # 16

        x6 = layers.UpSampling2D(size=2)(x6)       # 16 --> 32
        x6 = layers.Conv2D(in_filters, kernel_size=2, strides=1,
                           padding='same',
                           activation=None)(x6)    # 32
        x6 = layers.concatenate([x6, x1], axis=-1)
        x6 = layers.Conv2D(in_filters, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x6)  # 32
        x6 = layers.Conv2D(in_filters, kernel_size=3, strides=1,
                           padding='same',
                           activation='relu')(x6)  # 32

        deformation = layers.Conv2D(out_channels, kernel_size=1, strides=1,
                                    padding='same')(x6)  # 32

        nb, nh, nw, nc = tf.shape(deformation)

        # Regular grid.
        grid = regular_grid_2d(nh, nw)  # shape (H, W, 2)
        grid = tf.expand_dims(grid, axis=0)  # shape (1, H, W, 2)
        multiples = tf.stack([nb, 1, 1, 1])
        grid = tf.tile(grid, multiples)

        # Compute the new sampling grid.
        grid_new = grid + deformation
        grid_new = tf.clip_by_value(grid_new, -1, 1)

        # Sample the moving image using the new sampling grid.
        moved = grid_sample_2d(moving, grid_new)

        model = tf.keras.Model(inputs={'moving': moving, 'static': static},
                               outputs=moved, name='simple_fcn_2d')
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                      loss_weights=loss_weights)
        self.model = model

    def compile(self, optimizer='adam', loss=None, metrics=None,
                loss_weights=None):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                           loss_weights=loss_weights)

    def summary(self):
        return self.model.summary()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
            callbacks=None, validation_split=0.0, validation_data=None,
            shuffle=True, initial_epoch=0, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None,
            validation_freq=1, max_queue_size=10, workers=1,
            use_multiprocessing=False):
        return self.model.fit(x=x, y=y, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              callbacks=callbacks,
                              validation_split=validation_split,
                              validation_data=validation_data, shuffle=shuffle,
                              initial_epoch=initial_epoch,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_batch_size=validation_batch_size,
                              validation_freq=validation_freq,
                              max_queue_size=max_queue_size, workers=workers,
                              use_multiprocessing=use_multiprocessing)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1,
                 steps=None, callbacks=None, max_queue_size=10, workers=1,
                 use_multiprocessing=False, return_dict=False):
        return self.model.evaluate(x=x, y=y, batch_size=batch_size,
                                   verbose=verbose, steps=steps,
                                   callbacks=callbacks,
                                   max_queue_size=max_queue_size,
                                   workers=workers,
                                   use_multiprocessing=use_multiprocessing,
                                   return_dict=return_dict)

    def predict(self, x, batch_size=None, verbose=0,
                steps=None, callbacks=None, max_queue_size=10, workers=1,
                use_multiprocessing=False):
        return self.model.predict(x=x, batch_size=batch_size,
                                  verbose=verbose, steps=steps,
                                  callbacks=callbacks,
                                  max_queue_size=max_queue_size,
                                  workers=workers,
                                  use_multiprocessing=use_multiprocessing)

    def save_weights(self, filepath, overwrite=True):
        self.model.save_weights(filepath=filepath, overwrite=overwrite,
                                save_format=None)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
