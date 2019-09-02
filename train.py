#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:15:28 2019

@author: Purnendu Mishra
"""

#import tensorflow as tf
#from tensorflow.keras.backend import set_session
########################################################
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#set_session(tf.Session(config=config))
########################################################

from keras.optimizers import SGD, Adam
from keras.callbacks  import ModelCheckpoint
from keras.losses     import mean_squared_error

from models.yolse_model import YOLSEModel

from dataloader.CustomDataloaderYOLSE import DataAugmentor
from CustomCallback                   import CSVLogger, PolyLR

from pathlib import Path
import numpy as np

import argparse
import matplotlib.pyplot as plt

#%%****************************************************************************
parser = argparse.ArgumentParser()

parser.add_argument('-b','--batch_size', default=32, type=int, 
                    help='Batch size to be used for training')

parser.add_argument('-lr','--learning_rate', default=1e-2, type=float, 
                    help='Batch size to be used for training')

parser.add_argument('-e','--epochs', default = 100, type=int, 
                    help='Defines the number of epochs the model will be trained')

parser.add_argument('-o','--optimizer', default= 'SGD',type=str,
                    help= 'Defines optimizer used for trainign. Choose from "SGD", "adam" and "rmsprop"')

args = parser.parse_args()


#%%****************************************************************************
root       = Path.home()/'Documents'/'DATASETS'/'EgoGesture'

train_file = Path.cwd()/'train_files.csv'
val_file   = Path.cwd()/'val_files.csv'

#%%****************************************************************************
batch_size  = args.batch_size
num_epochs  = args.epochs
initial_lr  = args.learning_rate

target_size = (128, 128)
image_shape = target_size + (3,)

power       = 0.9 

opt         = args.optimizer

#%%****************************************************************************
exp_no      = '03-(5x5_kernel)'

experiment_name = "EXP-{}_YOLSE_fingertip_detect_BS{}_E{}".format(exp_no, batch_size, num_epochs) 

print(experiment_name)

#%%

records           = Path.cwd()/'records'
checkpoint_path   = Path.cwd()/'checkpoints'
final_weights     = Path.cwd()/'final_weights'

if not records.exists():
    records.mkdir(parents=True)
    
if not checkpoint_path.exists():
    checkpoint_path.mkdir(parents=True)
    
if not final_weights.exists():
    final_weights.mkdir(parents=True)
    

train_loader = DataAugmentor(rescale=1/255.0,
                             horizontal_flip = False,
                             vertical_flip   = False)

val_loader   = DataAugmentor(rescale=1/255.0)

train_data   = train_loader.flow_from_directory(root        = root,
                                                csv_file    = train_file,
                                                target_size = target_size,
                                                batch_size  = batch_size,
                                                shuffle     = True) 

val_data     = val_loader.flow_from_directory(root          = root,
                                              csv_file      = val_file,
                                              target_size   = target_size,
                                              batch_size    = batch_size,
                                              shuffle       = False)

steps_per_epoch   = len(train_data)
validation_steps  = len(val_data)  
#%%****************************************************************************
csvlog = CSVLogger(records/'{}.csv'.format(experiment_name), separator=',', append=False)

# Save the weights periodically
checkpoints = ModelCheckpoint('{}/{}_'.format(checkpoint_path, experiment_name) + '{val_mean_absolute_error:5f}.hdf5',
                              monitor = 'val_mean_absolute_error',
                              verbose = 1,
                              save_best_only = True, 
                              period = 1)



# Poly learning rate ploci for faster convergence
lrSchedule = PolyLR(base_lr   = initial_lr, 
                    power     = power,
                    nb_epochs = num_epochs,
                    steps_per_epoch=steps_per_epoch)

callbacks  = [csvlog, lrSchedule, checkpoints]


#%% The Model

model = YOLSEModel(input_shape = target_size + (3,))
#model.summary()

#%%

if opt == 'SGD':
    # Using SGD with momentum
    optimizer = SGD(lr=initial_lr, momentum=0.9, nesterov=True)

elif opt == 'adam':
    print('uisng ADAM optimizer')
    optimizer = Adam(lr=initial_lr, decay=5e-4)
#elif opt == 'rmsprop':
#    optimizer = RMSprop()

model.compile(loss      = mean_squared_error, 
              optimizer = optimizer,
              metrics   = ['mae'])


       
model.fit_generator(generator       = train_data,
                    validation_data = val_data,
                    steps_per_epoch = steps_per_epoch,
                    validation_steps= validation_steps,
                    epochs          = num_epochs,
                    verbose         = 1,
#                    workers         = 4,
                    callbacks       = callbacks,
                    use_multiprocessing = False
                   )


model.save('final_weights/{}_final-weights.h5'.format(experiment_name))
#%%****************************************************************************


#image, target1 = train_data[0]
#img  = image[0].astype(np.float32)
#
#target = target1[0]
#
#fig = plt.figure()
#plt.imshow(img)
#plt.show()
#plt.close(fig)
#
#for i in range(5):
#    fig1 = plt.figure(figsize=(15,15))
#    plt.subplot(1,5,i+1)
#    plt.imshow(target[:,:,i])
#plt.show()