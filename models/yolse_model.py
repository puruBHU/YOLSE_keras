#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:41:01 2019

@author: vlsilab
"""

import sys
sys.path.append('../')

from utility import conv_bn_relu

from keras.models import Model
from keras.layers import Input, Concatenate, Deconvolution2D,MaxPool2D, Conv2D

def YOLSEModel(input_shape = (128, 128, 3)):
    input_ = Input(shape = input_shape)
    
    x = conv_bn_relu(filters= 96,
                     name={'conv':'conv1', 'batch_norm':'BN1','activation':'act1'})(input_)
    x = MaxPool2D(name = 'pool_01')(x)
    
    x = conv_bn_relu(filters=96,
                      name={'conv':'conv2', 'batch_norm':'BN2','activation':'act2'})(x)
    x = MaxPool2D(name = 'pool_02')(x)
    
    x = conv_bn_relu(filters=128,
                      name={'conv':'conv3', 'batch_norm':'BN3','activation':'act3'})(x)
    
    upsample = Deconvolution2D(filters=128, kernel_size=(3,3), strides=(2,2), activation='relu',padding='same')(x)
    
    x = conv_bn_relu(filters=256,
                      name={'conv':'conv4', 'batch_norm':'BN4','activation':'act4'})(upsample)
    
    x = conv_bn_relu(filters=256,
                      name={'conv':'conv5', 'batch_norm':'BN5','activation':'act5'})(x)
    
    x = conv_bn_relu(filters=256,
                      name={'conv':'conv6', 'batch_norm':'BN6','activation':'act6'})(x)
    
    x = conv_bn_relu(filters=256,
                      name={'conv':'conv7', 'batch_norm':'BN7','activation':'act7'})(x)
    
    x = Concatenate()([x, upsample])
    
    x = conv_bn_relu(filters=128,
                      name={'conv':'conv8', 'batch_norm':'BN8','activation':'act8'})(x)
    
    x = Conv2D(filters = 5, kernel_size=(3,3), padding='same', activation='sigmoid')(x)
    
   
    return Model(inputs = input_, outputs = x)



if __name__=='__main__':
    model = YOLSEModel(input_shape=(128, 128, 3))
    model.summary()