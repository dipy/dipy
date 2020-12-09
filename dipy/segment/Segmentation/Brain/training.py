# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 04:09:58 2020

@author: Siddhesh
"""

#Importing the required libraries
import numpy as np
import os
import tensorflow as tf
import neural_net as nn

#Changing the directory
os.chdir("./New_Data")

#Setting environment variables
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICEs'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#Clearing tensorflow session
tf.keras.backend.clear_session()

#Loading the train data
x_train_1 = np.load("preprocessed_train_data_1.npy", allow_pickle = True)
x_train_2 = np.load("preprocessed_train_data_2.npy", allow_pickle = True)
x_train_3 = np.load("preprocessed_train_data_3.npy", allow_pickle = True)

#Loading the labels
y_train_1 = np.load("extracted_train_label_1.npy", allow_pickle = True)
y_train_2 = np.load("extracted_train_label_2.npy", allow_pickle = True)
y_train_3 = np.load("extracted_train_label_3.npy", allow_pickle = True)

#Loading the test data
x_test_1 = np.load("preprocessed_test_data_1.npy", allow_pickle = True)
x_test_2 = np.load("preprocessed_test_data_2.npy", allow_pickle = True)

#Combining the data and the labels
train_data = np.concatenate((x_train_1, x_train_2, x_train_3), axis = 0)
train_labels = np.concatenate((y_train_1, y_train_2, y_train_3), axis = 0)
test_data = np.concatenate((x_test_1, x_test_2), axis = 0)

#Displaying the respective shapes of the train data, train labels and test data
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)

#Displaying completion message
print("Done")

#Creating a Mirrored Strategy.
strategy = tf.distribute.MirroredStrategy()

#Displaying the number of gpus
print('Number of gpus: {}'.format(strategy.num_replicas_in_sync))

#Creating a model with learning rate 0.00003 using Adam Optimizer
model = nn.UNet3D(LR = (3 * 10e-5), opt = "adam")

#Printing the summary of the model
print(model.summary())

#Fitting the model
model.fit(train_data, train_labels, batch_size = 20, epochs = 50, validation_split = 0.2)

#Saving the model
model.save('brain_model.h5')

#Clearing tensorflow session
tf.keras.backend.clear_session()