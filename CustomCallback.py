#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 12:44:32 2018

@author: Purnendu Mishra
"""
#from __future__ import division
from keras.callbacks import Callback
from keras import backend as K


import numpy as np
import os
import csv
import six
import math

from collections import OrderedDict
from collections import Iterable
import matplotlib.pyplot as plt

class CSVLogger(Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.
    # Example
    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```
    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
       
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)
            


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # custom line
        # Addded to  get learning rate after each epoch
        logs.update({'lr':K.get_value(self.model.optimizer.lr)})

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())
#            print (logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
                
        row_dict = OrderedDict({'epoch': epoch, 'lr':K.get_value(self.model.optimizer.lr)})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
        

            
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())  


class PolyLR(Callback):
    def __init__(self, base_lr = 0.01, power = 0.9,
                 step_size = 5,
                 nb_epochs = None, 
                 steps_per_epoch = None, 
                 scale_factor = None,
                 fraction     = [0.65],
                 drop = 0.80,
                 mode = None):
        super(PolyLR, self).__init__()
        
        self.base_lr = base_lr
        self.power   = power
        self.scale_factor = scale_factor
        self.coefficent = 1.0
        self.fraction   = fraction
        self.drop       = drop
        
        self.cycle = step_size * steps_per_epoch
        self.max_iterations = steps_per_epoch * nb_epochs
        
         
        self.next_restart = self.cycle / 2.0
        
        self.iteration = 0
        self.count     = 0
        self.record    = 0 # keep updated after each update of fraction restart
        self.index     = 0
        self.batches_before_restart = 0
        self.mode = mode
        
        self.fraction   = fraction
        
        if not isinstance(self.fraction, list):
            raise TypeError('Expected fraction to be a list')
        
        if sum(self.fraction) != 1.0:
            self.fraction.append(1.0 - sum(self.fraction))
        
        self.history = {}
        
    def poly_lr(self):
        
        polylr = self.base_lr * ((1.0 - self.iteration / float(self.max_iterations)) ** self.power)
        
        if self.mode == 'sinusoidal':
            m = 1.0 * (self.scale_factor + math.sin(math.pi * (self.iteration / self.cycle)))
            return polylr * m
        
        elif self.mode == 'warm_restart':
            fraction_to_restart = self.batches_before_restart / float(self.cycle)
            m = (1.0 / self.scale_factor)  * (self.scale_factor + np.cos(2 * np.pi * fraction_to_restart))
            return polylr * m
        
        elif self.mode == 'triangular':
            c = np.floor((1 + self.iteration) / (2 * self.cycle))
            m = ((self.iteration) / float(self.cycle)) - (2 * c) - 1
            f = (1.0 - np.abs(m) + self.scale_factor) / self.scale_factor
            
            return f * polylr
        
        elif self.mode == 'poly_restart':
            return 1e-4 + self.base_lr * self.coefficent * ((1.0 - self.count / self.fractional_iteration) ** self.power)
        
        else:
            return polylr
    
    def on_train_begin(self, logs=None):
        
        self.fractional_iteration = np.ceil(self.fraction[self.index] * self.max_iterations)
        
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.poly_lr())
    
    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        
        self.iteration += 1
        self.count     += 1
        self.batches_before_restart += 1
        self.fractional_iteration = np.ceil(self.fraction[self.index] * self.max_iterations)
        
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)
        self.history.setdefault('batches_before_restart',[]).append(self.batches_before_restart)
        
        
        # Set the new value of Learning after each batch
        K.set_value(self.model.optimizer.lr, self.poly_lr())
        
        if self.iteration % self.next_restart == 0:
            self.batches_before_restart = 0
            
        if self.index < (len(self.fraction) - 1)  and self.iteration == self.fractional_iteration + self.record:
            self.index += 1 
            self.count = 0
            self.coefficent *= self.drop
            self.record = self.fractional_iteration + self.record
            
            
        
    def plot_lr(self):
        '''Helper function to quickly inspect learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.xlabel('Iterations')
        plt.ylabel('learning rate')
        plt.grid(True)
        plt.show()
          
