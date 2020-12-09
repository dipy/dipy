# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 01:36:55 2020

@author: Siddhesh
"""

#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib

#Specifying the data directories (Brain)
train_data_dir = "Brain_Data/Task01_BrainTumour/imagesTr"
train_label_dir = "Brain_Data/Task01_BrainTumour/labelsTr"
test_data_dir = "Brain_Data/Task01_BrainTumour/imagesTs"
  
#Creating a function to load data
def load_data(data_directory):
    
    #Creating an empty array to store the data
    data_arr = []
    dir_names = []

    #Iterating through the directory
    for directory in os.listdir(data_directory):
        
        #Replacing the unwanted characters
        directory = directory.replace('._', '')
        
        #Checking whether the directory has been repeated
        if(directory in dir_names):
            
            #Continuing to the next iteration
            continue
        
        #Adding the directory names to the array
        dir_names.append(directory)

        #Specifying the file path
        path = os.path.join(data_directory + "/" + directory)
        
        #Trying the operation
        try:

            #Displaying the file name
            print(os.path.basename(path))
            
            #Loading the data
            data = nib.load(path)
            
        #Handling the exception
        except Exception as e:
            
            #Displaying error message
            print("Unable to load current image!", e)
            
            #Continuing to the next iteration
            continue
        
        #Storing the data in the list
        data_arr.append(data)
        
    #Returning the list
    return np.array(data_arr)

#Creating a function to display data in 2 dimensions
def display_2d(image, last_index = 0):
    
    #Accessing the pixels
    f_image = image.get_fdata()
    
    #Retrieving the shape of the image
    shape = f_image.shape
    
    #Checking whether the user gives an index out of range
    if(last_index >= shape[2]):
        
        #Displaying error message
        print("The value of last index cannot be", shape[2], "or more!")
        
        #Returning nothing
        return
    
    #Displaying the image
    plt.imshow(f_image[:,:,last_index], aspect = 0.5)
    
    #Returning nothing
    return
        
#Loading the data
train_data = load_data(train_data_dir)
train_labels = load_data(train_label_dir)
test_data = load_data(test_data_dir)

#Checking the shape of the data and labels
print("Printing the shapes of the data and labels")
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print("")

#Saving the data
np.save("train_data.npy", train_data, allow_pickle = True)
np.save("train_labels.npy", train_labels, allow_pickle = True)
np.save("test_data.npy", test_data, allow_pickle = True)