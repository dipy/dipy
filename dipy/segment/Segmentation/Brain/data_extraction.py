# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Referred https://www.tutorialspoint.com/How-are-files-extracted-from-a-tar-file-using-Python#:~:text=You%20can%20use%20the%20tarfile,method%20of%20the%20tarfile%20module.

#Importing the required libraries
import tarfile

#Creating a function to extract data from a tar file
def extract_data(file, output_path):

    #Opening the required tar file
    my_tar = tarfile.open(file)
    
    #Extracting the data from the tar file in a specified folder
    my_tar.extractall(output_path)
    
    #Closing the tar file
    my_tar.close()
    
    #Displaying completion message
    print("Files successfully extracted!")
    
    #Returning nothing
    return

#Specifying the tar file
brain_file = 'Task01_BrainTumour.tar'

#Specifying the output path
brain_path = './Brain_Data'

#Extracting data from the tar file
extract_data(file = brain_file, output_path = brain_path)