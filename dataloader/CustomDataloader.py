#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:04:11 2019

@author: Purnendu Mishra

Description:
    Including the index value of anchor which is nearest to the given fingertip
    Index data is converted into categorical form
     
    Adding Data augmentation:
        1. random horizontal flip
        2. random vertical flip
"""

# Standard library import
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from skimage.io import imread

# Keras library import
from tensorflow.keras import backend as K
from keras.utils import Sequence, to_categorical


def Gauss2D(sigma = 1.5, coordinate = None):
    x,y  = coordinate
    
    j,i  = np.ogrid[0:64,0:64]
    
    a    = (2 * np.square(sigma))
    
    b    = 1.0 / (a * np.pi)
    
    gaussian = np.exp( - (np.square(i - x) + np.square(j - y)) / a)
    return gaussian

def CropImage(image = None, bndbox=None):
    
    h, w = image.shape[:2]
    
    xtop = np.int16(w * bndbox[0])
    ytop = np.int16(h * bndbox[1])
    
    xbot = np.int16(w * bndbox[2])
    ybot = np.int16(h * bndbox[3])
    
    cropped_image  = image[ytop:ybot, xtop:xbot,:]
    
    return cropped_image

def load_image(path = None, target_size = (128,128), bndbox = None):
    """Load an image as an numpy array
    
    Arguments:
        path: Path to the image file
        target_size:  Either , None (default to original size),
                      int or tuple (width, height)
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    else:
        target_size = target_size
            
    image = imread(path)
    
    # Crop the hand area from the whole image
    image = CropImage(image  = image, bndbox = bndbox)
    # resize the image
    if target_size:
        image = cv2.resize(image, (target_size[0], target_size[1]), 
                           interpolation = cv2.INTER_CUBIC)
    
    return image


         
def PreProcessGT(gt, target_size = (128, 128)):
    l = len(gt)
    
    h, w = target_size
    
    l = int(len(gt) / 2)
    ground_truth = []
    
    for i  in range(l):
        x = gt[2 * i]
        y = gt[2 * i + 1]
    
#        x = np.int16(w * x)
#        y = np.int16(h * y)
        
        ground_truth.append([x,y])
        
    return np.array(ground_truth, dtype= np.float32)


def encoder(gt):
    
    non_zero_index = np.nonzero(gt[:,0])[0]
    
    fingermap = np.zeros(shape=(64,64,5),dtype=np.float32)
    h, w      = fingermap.shape[:2]
    
    for i in non_zero_index:
        x, y = gt[i]
   
        
        x = w * x
        y = h * y
        
        fingermap[:,:,i] = Gauss2D(coordinate = (x,y))

    return fingermap

    
class DataAugmentor(object):
    
    def __init__(self,
                rotation_range   = 0.,
                zoom_range       = 0.,
                horizontal_flip  = False,
                vertical_flip    = False,
                rescale          = None,
                data_format      = None,
                normalize = False,
                mean = None,
                std = None 
                ):
        
        if data_format is None:
            data_format = K.image_data_format()
            
        self.data_format = data_format
        
        if self.data_format == 'channels_last':
            self.row_axis = 0
            self.col_axis = 1
            self.channel_axis = 2
            
        self.rotation_range  = rotation_range
        self.zoom_range      = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip   = vertical_flip
        self.normalize       = normalize
        
        self.rescale = rescale
        self.mean = mean
        self.std = mean
        
        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
            
        elif len(zoom_range) == 2:
            self.zoom_range =[zoom_range[0], zoom_range[1]]
            
        else:
            raise ValueError("""zoom_range should be a float or
                             a tuple or lis of two floats. 
                             'Receved args:'""", zoom_range)
            
        
    def flow_from_directory(self,
                           root        = None, 
                           csv_file    = None,
                           target_size = (128,128),
                           batch_size  = 8,
                           shuffle     = False,
                           data_format = None
                           ):
        return Dataloader(
                root,
                csv_file,
                self,
                target_size = target_size,
                batch_size  = batch_size,
                shuffle     = shuffle,
                data_format = self.data_format,
                
                 )
    def random_transforms(self, samples, seed=None):
        
        if seed is not None:
            np.random.seed(seed)
            
        if len(samples) != 2:
            x = samples
            y = None
            
        else:
            x = samples[0]
            y = samples[1]
        
                    
#        if self.rotation_range:
#            theta = int(180 * np.random.uniform(-self.rotation_range,
#                                                self.rotation_range))
#            
#            (h, w) = x.shape[:2]
#            (cx, cy) = [w//2, h//2]
#            
#            M = cv2.getRotationMatrix2D((cx,cy), -theta, 1.0)
#            x = cv2.warpAffine(x , M, (w,h))
#            y = cv2.warpAffine(y,  M, (w,h))
            
            
        if self.horizontal_flip:    
            if np.random.random() < 0.5:
                x      = x[:,::-1,:] #flip x along x axis
                
                # To flip the fingertip coordinate horizontally
                # subtract x coordinates from 1.0
                non_zeros_gt      = np.nonzero(y[:,0])
                y[non_zeros_gt,0] = 1.0 - y[non_zeros_gt,0]
                
                
        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = x[::-1,:,:]
              
                # To flip the fingertip coordinate vertically
                # subtract y coordinates from 1.0
                non_zeros_gt      = np.nonzero(y[:,1])
                y[non_zeros_gt,1] = 1.0 - y[non_zeros_gt,1]
           
        
#        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
#            zx, zy = 1, 1
#        else:
#            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
#            
#        x = image.random_zoom(x, (zx,zy), channel_axis=self.channel_axis)
#        y = image.random_zoom(y, (zx,zy), channel_axis=self.channel_axis)
        
        return (x,y)
    
    def standardize(self,x):
        """Apply the normalization configuration to a batch of inputs.
            Arguments:
                x: batch of inputs to be normalized.
            Returns:
                The inputs, normalized.
        """       
        x = x.astype('float32')
        
        if self.rescale:
            x *= self.rescale
            
            
        if self.normalize:
            if self.mean is not None:
                x -= self.mean
            else:
                x -= np.mean(x, axis=self.channel_axis, keepdims=True)
                
                
            if self.std is not None:  
                x /= self.std
                
            else:
                x /= (np.std(x, axis=self.channel_axis, keepdims=True) + 1e-7)

        return x
            

class Dataloader(Sequence):
    
    def __init__(self, 
                 root        = None, 
                 csv_file    = None, 
                 image_data_generator = None,
                 target_size = (128,128),
                 batch_size  = 32, 
                 shuffle     = False,
                 data_format = 'channels_last'):
        
#        super(Dataloader, self).__init__(self)
        
        if data_format is None:
            data_format = K.image_data_format()
        
        self.root  =  root
        self.files = pd.read_csv(csv_file, header=None)
        
        self.target_size =  target_size
        self.batch_size  =  batch_size
        
        self.data_format =  data_format
        self.shuffle     =  shuffle
        
        self.image_data_generator = image_data_generator
        
        if data_format == 'channels_last':
            self.row_axis        = 1
            self.col_axis        = 2
            self.channel_axis    = 3
            self.image_shape     = self.target_size + (3,)
            
        self.on_epoch_end()    
    
    def __len__(self):
        return int(np.ceil(len(self.files) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        # total number of samples in the dataset
        n = len(self.files)
     
        if n > idx * self.batch_size:
            current_batch_size = self.batch_size
        else:
            current_batch_size = n - self.batch_size
        
        image_batch  = self.f.iloc[idx * current_batch_size : (idx + 1) * current_batch_size, 0].values
        bndbox_batch = self.f.iloc[idx * current_batch_size : (idx + 1) * current_batch_size, 1:5].values
        label_batch  = self.f.iloc[idx * current_batch_size : (idx + 1) * current_batch_size, 11:].values
        
        batch_x = []
        batch_y = []

            
        for m, files in enumerate(image_batch):
            
            # The full path of the image
            image_path = self.root/files
            bndbox     = bndbox_batch[m]
            
            x = load_image(path = image_path, target_size = self.target_size, bndbox = bndbox)
            
            # Normalize the image
            x = self.image_data_generator.standardize(x)
           
            ground_truth = PreProcessGT(label_batch[m])
            
            # Data Augmentation
            x, ground_truth = self.image_data_generator.random_transforms((x, ground_truth))
            
            ground_truth = encoder(gt = ground_truth)
            
            x = np.array(x, dtype=np.float32)
            y = np.array(ground_truth, dtype=np.float32)
            
            batch_x.append(x)
            batch_y.append(y)
         
            
            # print(y.shape)
            
        
        batch_x = np.array(batch_x, dtype = np.float32)
        batch_y = np.asarray(batch_y)
            
        return batch_x, batch_y
    
    def on_epoch_end(self):
        'Shuffle the at the end of every epoch'
        self.f = self.files.copy()

        if self.shuffle == True:
            self.f  = self.f.reindex(np.random.permutation(self.f.index))
            
            
    
