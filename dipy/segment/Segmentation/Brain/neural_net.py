# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 04:35:54 2020

@author: Siddhesh
"""

#Importing required libraries
import tensorflow as tf
import loss_functions as lf

#Creating a 3D UNet architecture
def UNet3D(LR, opt = "adam"):

    #Defining the input
    inputs = tf.keras.layers.Input((80, 90, 78, 4))

    #CONTRACTION PATH
    
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c1 = tf.keras.layers.Conv3D(4, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(inputs)
    #Adding a Dropout Layer with a rate of 0.1
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c1 = tf.keras.layers.Conv3D(4, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c1)
    #Adding a Pooling Layer with a pool size 3x3x3
    p1 = tf.keras.layers.MaxPooling3D(pool_size = (2, 2, 2))(c1)

    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c2 = tf.keras.layers.Conv3D(8, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p1)
    #Adding a Dropout Layer with a rate of 0.1
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c2 = tf.keras.layers.Conv3D(8, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c2)
    #Adding a Pooling Layer with a pool size 3x3x3
    p2 = tf.keras.layers.MaxPooling3D(pool_size = (2, 2, 2))(c2)

    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c3 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p2)
    #Adding a Dropout Layer with a rate of 0.1
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c3 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c3)
    #Adding a Pooling Layer with a pool size 3x3x3
    p3 = tf.keras.layers.MaxPooling3D(pool_size = (2, 2, 2))(c3)

    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c4 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p3)
    #Adding a Dropout Layer with a rate of 0.1
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c4 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c4)
    #Adding a Pooling Layer with a pool size 3x3x3
    p4 = tf.keras.layers.MaxPooling3D(pool_size = (2, 2, 2))(c4)

    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c5 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p4)
    #Adding a Dropout Layer with a rate of 0.1
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c5 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c5)

    #EXPANSIVE PATH
    
    #Adding a 3D Inverse Convolution Layer with x convolution filters and kernel size 2x2x2
    u6 = tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), strides = (2, 2, 2), padding = 'same')(c5)
    #Padding the 3rd dimension with 0's on one side
    u6 = tf.keras.layers.ZeroPadding3D(padding = ((0, 0), (0, 1), (0, 1)))(u6)
    
    #Concatenating the 6th layer with the 4th layer
    u6 = tf.keras.layers.concatenate([u6, c4])
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c6 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u6)
    #Adding a Dropout Layer with a rate of 0.2
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c6 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c6)
    
    #Adding a 3D Inverse Convolution Layer with x convolution filters and kernel size 2x2x2
    u7 = tf.keras.layers.Conv3DTranspose(16, (2, 2, 2), strides = (2, 2, 2), padding = 'same')(c6)
    #Padding the 3rd dimension with 0's on one side
    u7 = tf.keras.layers.ZeroPadding3D(padding = ((0, 0), (0, 0), (0, 1)))(u7)
    #Concatenating the 7th layer with the 3rd layer
    u7 = tf.keras.layers.concatenate([u7, c3])
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c7 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u7)
    #Adding a Dropout Layer with a rate of 0.2
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c7 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c7)

    #Adding a 3D Inverse Convolution Layer with x convolution filters and kernel size 2x2x2
    u8 = tf.keras.layers.Conv3DTranspose(8, (2, 2, 2), strides = (2, 2, 2), padding = 'same')(c7)
    #Padding the 3rd dimension with 0's on one side
    u8 = tf.keras.layers.ZeroPadding3D(padding = ((0, 0), (1, 0), (0, 1)))(u8)
    #Concatenating the 8th layer with the 2nd layer
    u8 = tf.keras.layers.concatenate([u8, c2])
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c8 = tf.keras.layers.Conv3D(8, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u8)
    #Adding a Dropout Layer with a rate of 0.2
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c8 = tf.keras.layers.Conv3D(8, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c8)

    #Adding a 3D Inverse Convolution Layer with x convolution filters and kernel size 2x2x2
    u9 = tf.keras.layers.Conv3DTranspose(4, (2, 2, 2), strides = (2, 2, 2), padding = 'same')(c8)
    #Padding the 3rd dimension with 0's on one side
    u9 = tf.keras.layers.ZeroPadding3D(padding = ((0, 0), (0, 0), (0, 0)))(u9)
    #Concatenating the 9th layer with the 1st layer
    u9 = tf.keras.layers.concatenate([u9, c1])
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c9 = tf.keras.layers.Conv3D(4, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u9)
    #Adding a Dropout Layer with a rate of 0.2
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 3x3x3
    c9 = tf.keras.layers.Conv3D(4, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c9)
    
    #Adding a 3D Convolution Layer with x convolution filters and kernel size 1x1x1 and sigmoid activation function
    outputs = tf.keras.layers.Conv3D(1, (1, 1, 1), activation = 'sigmoid')(c9)

    #Building the model using the inputs and outputs
    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    
    #Training on multiple gpus
    parallel_model = tf.keras.utils.multi_gpu_model(model, gpus = 4)
    
    #Checking whether optimizer is Adam
    if(opt == "adam"):
        
        #Using Adam Optimizer
        opt = tf.keras.optimizers.Adam(learning_rate = LR)
        
    #Checking whether optimizer is SGD
    elif(opt == "sgd"):
        
        #Using SGD Optimizer
        opt = tf.keras.optimizers.SGD(learning_rate = LR)
    
    #Compiling the model
    parallel_model.compile(optimizer = opt, loss = lf.bce_dice_loss, metrics = [lf.accuracy])

    #Returning the model
    return parallel_model