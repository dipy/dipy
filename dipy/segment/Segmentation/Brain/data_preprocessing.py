# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 03:53:16 2020

@author: Siddhesh
"""

#Importing the required libraries
import numpy as np
from scipy.ndimage import zoom

#Loading the data
train_data = np.load("train_data.npy", allow_pickle = True)
train_labels = np.load("train_labels.npy", allow_pickle = True)
test_data = np.load("test_data.npy", allow_pickle = True)

#Creating a function to preprocess an image
def preprocess(data):
    
    #Creating an empty array to store the images
    arr = []
    
    #Defining a counter to keep track of the number of images that have been preprocessed
    count = 0
    
    #Iterating through the data
    for img in data:
    
        #Retrieving the image
        f_image = img.get_fdata()
        
        #Resizing the image
        f_image = zoom(f_image, (0.5, 0.5, 0.5, 1))
        
        #Cropping the image to include only the region of the tumors
        f_image = f_image[20:100, 10:100, :]
        
        #Normalizing the image
        f_image = f_image/np.max(f_image)
           
        #Converting the image to a float
        f_image = f_image.astype('float32')
        
        #Storing the image in the array
        arr.append(f_image)
        
        #Incrementing the counter
        count += 1
        
        #Displaying the current count
        print(count)
        
    #Returning the array
    return np.array(arr)

#Creating a function to extract the labels
def extract_labels(labels):
    
    #Creating an empty array to store the images
    arr = []
    
    #Defining a counter to keep track of the number of images that have been preprocessed
    count = 0
    
    #Iterating through the data
    for img in labels:
    
        #Retrieving the image
        f_image = img.get_fdata()
    
        #Resizing the image
        f_image = zoom(f_image, (0.5, 0.5, 0.5))
        
        #Cropping the image to include only the region of the tumors
        f_image = f_image[20:100, 10:100, :]
        
        #Storing the image in the array
        arr.append(f_image)
        
        #Incrementing the counter
        count += 1
        
        #Displaying the current count
        print(count)
        
    #Returning the array
    return np.array(arr)

#Preprocessing the train data in parts to avoid memory exhaustion
preprocessed_train_data_1 = preprocess(train_data[:len(train_data)//3])
preprocessed_train_data_2 = preprocess(train_data[len(train_data)//3:(2 * len(train_data)//3)])
preprocessed_train_data_3 = preprocess(train_data[(2 * len(train_data)//3):])

#Displaying the respective shapes of the different parts of the train data
print(preprocessed_train_data_1.shape)
print(preprocessed_train_data_2.shape)
print(preprocessed_train_data_3.shape)
    
#Saving the train data in parts to avoid memory exhaustion
np.save("preprocessed_train_data_1.npy", preprocessed_train_data_1, allow_pickle = True)
np.save("preprocessed_train_data_2.npy", preprocessed_train_data_2, allow_pickle = True)
np.save("preprocessed_train_data_3.npy", preprocessed_train_data_3, allow_pickle = True)
    
#Preprocessing the test data in parts to avoid memory exhaustion
preprocessed_test_data_1 = preprocess(test_data[:len(test_data)//2])
preprocessed_test_data_2 = preprocess(test_data[len(test_data)//2:])

#Displaying the respective shapes of the different parts of the test data
print(preprocessed_test_data_1.shape)
print(preprocessed_test_data_2.shape)
    
#Saving the test data in parts to avoid memory exhaustion
np.save("preprocessed_test_data_1.npy", preprocessed_test_data_1, allow_pickle = True)
np.save("preprocessed_test_data_2.npy", preprocessed_test_data_2, allow_pickle = True)
    
#Extracting the train labels in parts to avoid memory exhaustion
extracted_train_label_1 = extract_labels(train_labels[:len(train_labels)//3])
extracted_train_label_2 = extract_labels(train_labels[len(train_labels)//3:(2 * len(train_labels)//3)])
extracted_train_label_3 = extract_labels(train_labels[(2 * len(train_labels)//3):])
    
#Displaying the respective shapes of the different parts of the train labels
print(extracted_train_label_1.shape)
print(extracted_train_label_2.shape)
print(extracted_train_label_3.shape)

#Saving the train labels in parts to avoid memory exhaustion
np.save("extracted_train_label_1.npy", extracted_train_label_1, allow_pickle = True)
np.save("extracted_train_label_2.npy", extracted_train_label_2, allow_pickle = True)
np.save("extracted_train_label_3.npy", extracted_train_label_3, allow_pickle = True)