"""
========================
Deep-learning-based 2D Deformable Image Registration
========================

This example explains how to use the ``dipy.nn.registration`` module to perform
2D deformable image registration using deep learning. We will train a
convolutional neural network to register handwritten digit images from the
MNIST dataset.

The deep learning model, given a pair of the static and the moving images as
input, computes the pixel-wise deformation between the two images. This
deformation field also called a dense deformation field or registration field,
is a set of sampling points where the moving image needs to be sampled to
align it to the static image. Within the deep learning model, the moving
image is resampled at the new locations and the transformed image is
outputted.

In this example, we will show how to

- load and compile the deep learning model.
- write the data loaders and prepare the data to feed to the model.
- train and evaluate the model.
- register new images using the trained model.
- save and load the weights (parameters) of the trained model.
"""

import numpy as np
import scipy.ndimage
import math
from dipy.nn.registration import FCN2d
from dipy.nn.metrics import normalized_cross_correlation_loss
from distutils.version import LooseVersion
from dipy.utils.optpkg import optional_package
plt, _, _ = optional_package("matplotlib.pyplot")
tf, have_tf, _ = optional_package('tensorflow')
if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')

"""
To get started, we will load the MNIST dataset into the memory and filter 
the training and testing sets to keep just one class of digits. We can 
change the ``label`` variable to select a different class.
"""

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Discard other digits.
label = 7  # Which digit images to keep.
x_train = x_train[y_train == label].copy()  # shape (n_train, 28, 28)
x_test = x_test[y_test == label].copy()  # shape (n_test, 28, 28)

# We'll also resize the images to (32, 32), just to keep it a power of 2.
x_train = scipy.ndimage.zoom(x_train, (1, 32/28, 32/28))
x_test = scipy.ndimage.zoom(x_test, (1, 32/28, 32/28))

"""
Let's select our static image randomly from the test set. We'll also select 
few images (``x_sample``) for visualizing the results.
"""

idx = np.random.randint(x_test.shape[0])
static = x_test[idx].copy()

num_samples = 2  # Number of sample images to visualize.
# Sample images to show results.
idxs = np.random.choice(x_test.shape[0], replace=False,
                        size=num_samples)
x_sample = x_test[idxs].copy()  # shape (num_samples, 32, 32)

"""
Implement a data loader that fetches and preprocesses batches of images for 
real-time data feeding to our model. 
"""


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


"""
Let's define some hyperparameters used for training the network.
"""

batch_size = 32
epochs = 100
lr = 0.004  # learning rate

"""
Create the data loader objects for the training, testing and the 
sample sets.
"""

train_loader = DataLoader(x_train, static, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(x_test, static, batch_size=batch_size, shuffle=True)
sample_loader = DataLoader(x_sample, static, shuffle=True)

"""
After completing the data processing, we will instantiate and compile our 
deep learning model with the loss and the optimizer. In this example, 
we will use the normalized cross-correlation (NCC) loss and the stochastic 
gradient descent (SGD) as the optimizer. We also need to specify the input 
shape in the form of (height, width, channels). Here, it is equal (32, 32, 1).
"""

loss = normalized_cross_correlation_loss()
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

# Compile the model with the loss and the optimizer.
model = FCN2d(input_shape=(32, 32, 1), optimizer=optimizer, loss=loss)
# model.compile(loss=loss, optimizer=optimizer)  # this works too

"""
Now that everything is set up, we can go ahead and train our model by 
passing the training and testing data loaders to the ``fit`` method along 
with the number of epochs we want to train the model for. After the 
optimization is done, a history object is returned which records the 
training and validation losses for every epoch.
"""

hist = model.fit(train_loader, epochs=epochs, validation_data=test_loader)

"""
Let's plot the losses to see if the model converged. 
"""

plt.plot(hist.history['loss'], color='royalblue', label='Train')
plt.plot(hist.history['val_loss'], color='seagreen', label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.title('Loss vs. Epochs')
plt.show()
plt.savefig('loss_plot.png')

"""
.. figure:: loss_plot.png
   :align: center

    Training loss and testing loss w.r.t. epochs.   
"""

"""
Now let's evaluate the trained model on the testing and sample sets. And 
also compute the average NCC values for these sets. The NCC is just the 
negative of the NCC loss. The NCC values range from -1 to 1. For a perfect 
match, it is equal to 1 and for complete dissimilarity, it is -1. 
"""

test_loss = model.evaluate(test_loader)
sample_loss = model.evaluate(sample_loader)

test_ncc = -test_loss
sample_ncc = -sample_loss

print('Test NCC: ', test_ncc)
print('Sample NCC: ', sample_ncc)

"""
Now let's warp the sample images and see if they get similar to the static 
image. The returned registered images by the model are currently in the [0, 
1] range. So we need to perform some post-processing to convert them to the 
normal 8-bit images.
"""

moved = model.predict(sample_loader)  # shape (num_samples, 32, 32, 1)
# Note: 'predict' returns outputs as numpy arrays.

"""
Finally, let's visualize the transformed image together with the static and 
the moving images.
"""

moved = moved.squeeze(axis=-1)  # Remove the channel dim.
moved = moved * 255.0  # Rescale to [0, 255].
moved = moved.astype(np.uint8)  # Convert back to 8-bit images.

moving = x_sample.copy()  # shape (num_samples, 32, 32)

static_ = static[np.newaxis, ...]  # shape (1, 32, 32)
static_ = np.repeat(static_, repeats=moving.shape[0], axis=0)

# Plot images.
nb = moved.shape[0]
fig = plt.figure(figsize=(3 * 1.7, nb * 1.7))
titles_list = ['Static', 'Moved', 'Moving']
images_list = [static_, moved, moving]
for i in range(nb):
    for j in range(3):
        ax = fig.add_subplot(nb, 3, i * 3 + j + 1)
        if i == 0:
            ax.set_title(titles_list[j], fontsize=20)
        ax.set_axis_off()
        ax.imshow(images_list[j][i], cmap='gray')
plt.tight_layout()
plt.show()
plt.savefig('sample_results.png')

"""
.. figure:: sample_results.png
   :align: center

    Deformable image registration results.
"""

"""
After the training is finished we can save the parameters of the model by 
simply calling the ```save_weights``` method and specifying the path to the 
file that we want to save the weights to. 
"""

model.save_weights('fcn2d.h5')

"""
Restoring the model is also very easy. We first need to create an instance 
of the model class that we want to restore and then we can load the weights by 
simply calling the ```load_weights``` method providing the path to the saved 
weights file.
"""

model_new = FCN2d(input_shape=(32, 32, 1))
model_new.load_weights('fcn2d.h5')

"""
To evaluate or fine-tune the model, we need to compile it with the
same loss function that was used earlier to train. This can also be done by 
simply passing the loss function during the instantiation of the model.
"""

loss = normalized_cross_correlation_loss()
model_new.compile(loss=loss)

"""
Now let's evaluate our restored model and check if the losses for the test 
and the sample sets match with the values we got earlier.
"""

test_loss_new = model_new.evaluate(test_loader)
sample_loss_new = model_new.evaluate(sample_loader)

print('Diff: ', abs(test_loss_new-test_loss))
print('Diff: ', abs(sample_loss_new-sample_loss))
