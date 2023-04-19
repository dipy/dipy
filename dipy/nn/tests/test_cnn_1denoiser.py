import pytest
import numpy as np
from dipy.utils.optpkg import optional_package
tf, have_tf, _ = optional_package('tensorflow')
if have_tf:
    from dipy.nn.cnn_1d_denoising import Cnn1DDenoiser


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_default_Cnn1DDenoiser_sequential():
    # Create dummy data
    normal_img = np.random.rand(10, 10, 10, 30)
    nos_img = normal_img + np.random.normal(loc=0.0, scale=0.1,
                                            size=normal_img.shape)
    x = np.random.rand(10, 10, 10, 30)
    # Test 1D denoiser
    model = Cnn1DDenoiser(30)
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['accuracy'])
    epochs = 1
    hist = model.fit(nos_img, normal_img, epochs=epochs)
    data = model.predict(x)
    model.evaluate(nos_img, normal_img, verbose=2)
    _ = hist.history['accuracy'][0]
    assert data.shape == x.shape


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_default_Cnn1DDenoiser_flow(pytestconfig):
    # Create dummy data
    normal_img = np.random.rand(10, 10, 10, 30)
    nos_img = normal_img + np.random.normal(loc=0.0, scale=0.1,
                                            size=normal_img.shape)
    x = np.random.rand(10, 10, 10, 30)
    # Test 1D denoiser with flow API
    model = Cnn1DDenoiser(30)
    if pytestconfig.getoption('verbose') > 0:
        model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['accuracy'])
    epochs = 1
    hist = model.fit(nos_img, normal_img, epochs=epochs)
    _ = model.predict(x)
    model.evaluate(nos_img, normal_img, verbose=2)
    accuracy = hist.history['accuracy'][0]
    assert accuracy > 0
