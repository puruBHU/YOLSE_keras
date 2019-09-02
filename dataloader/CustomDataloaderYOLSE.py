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

import pandas as pd
import numpy as np

from pathlib import Path
from skimage.io import imread
import cv2

from tensorflow.keras import backend as K
from keras.utils import Sequence, to_categorical

def Gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h



def load_image(path, target_size = (128,128)):
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
    
    for i in non_zero_index:
        x, y = gt[i]
        
        x = np.int16(64 * x)
        y = np.int16(64 * y)
#        print(x.shape)
        
        # limit the top coordinate 
        xtop = np.maximum(0, int(x - 3))
        ytop = np.maximum(0, int(y - 3))
        
       
        # if xtop or ytop >  124
        xtop = np.minimum(xtop, 59)
        ytop = np.minimum(ytop, 59)
       
    
        fingermap[ytop : (ytop + 5), xtop :( xtop + 5), i] =  np.ones(shape=(5,5), dtype=np.float32)

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
        
        image_batch = self.f.iloc[idx * current_batch_size : (idx + 1) * current_batch_size, 0].values
        
        label_batch = self.f.iloc[idx * current_batch_size : (idx + 1) * current_batch_size, 7:].values
        
        batch_x = []
        batch_y = []
            
        for m, files in enumerate(image_batch):
            
            image_name = files.split('/')[-1]
            # The full path of the image
            image_path = self.root/'resized_224x224'/image_name
            
            x = load_image(image_path, target_size = self.target_size)
            
            # Normalize the image
            x = self.image_data_generator.standardize(x)
            ground_truth = PreProcessGT(label_batch[m])
            
            x, ground_truth = self.image_data_generator.random_transforms((x, ground_truth))
            
            ground_truth = PreProcessGT(gt =  label_batch[m])
            ground_truth = encoder(gt = ground_truth)
            
            x = np.array(x, dtype=np.float32)
            y = np.array(ground_truth, dtype=np.float32)
            
            batch_x.append(x)
            batch_y.append(y)
            
        batch_x = np.array(batch_x, dtype = np.float32)
        batch_y = np.array(batch_y, dtype = np.float32)
            
        return batch_x, batch_y
    
    def on_epoch_end(self):
        'Shuffle the at the end of every epoch'
        self.f = self.files.copy()

        if self.shuffle == True:
            self.f  = self.f.reindex(np.random.permutation(self.f.index))
            
            
    
