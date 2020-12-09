# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:56:06 2020

@author: Siddhesh
"""

"""
Referred 
https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
"""

#Importing required libraries
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy
import keras.backend as K

#Creating a function to calculate the dice accuracy
def accuracy(y_true, y_pred):
            
    #Adding smoothing factor
    smooth = 1.

    #Calculating the intersection
    intersection = K.sum(K.abs(y_true * y_pred), axis = -1)
    
    #Returning the dice coefficient
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    
#Creating a function to calculate dice loss
def dice_loss(y_true, y_pred):
    
    #Returning the loss
    return 1 - accuracy(y_true, y_pred)

#Creating a function to calculate bce dice loss
def bce_dice_loss(y_true, y_pred):
    
    #Combining the dice loss with binary crossentropy loss
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    
    #Returning the loss
    return loss

#Creating a function to calculate cce dice loss
def cce_dice_loss(y_true, y_pred):
    
    #Combining the dice loss with categorical crossentropy loss
    loss = categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    
    #Returning the loss
    return loss

#Creating a function to calculate the tversky accuracy
def t_accuracy(y_true, y_pred):
    
    #Adding smoothing factor
    smooth = 1.
    
    #Calculating the true positives
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    
    #Calculating the false negatives
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    
    #Calculating the false negatives
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    
    #Setting the value of alpha    
    alpha = 0.7
    
    #Returning the tversky coefficient
    return (true_pos + smooth)/(true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

#Creating a function to calculate tversky loss
def tversky_loss(y_true, y_pred):
    
    #Returning the loss
    return 1 - t_accuracy(y_true, y_pred)

#Creating a function to combine tversky and dice loss
def tversky_dice_loss(y_true, y_pred):
    
    #Combining the dice loss with tversky loss
    loss = tversky_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
    
    #Returning the loss
    return loss
    