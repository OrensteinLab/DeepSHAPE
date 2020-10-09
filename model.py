# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:39:29 2020
@author: Yifat
"""
import os
###############################################################################
#from keras.models import Sequential
#from keras.layers import *
###############################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
###############################################################################
def model(neighbours,num_features,loss_,optimizer_,activation_last_layer):
 
    model_fc = Sequential()
    model_fc.add(Flatten(input_shape=(neighbours+1,num_features)))
    model_fc.add(Dense(100, activation='relu'))
    model_fc.add(Dense(50, activation='relu'))
    model_fc.add(Dense(10, activation='relu'))
    model_fc.add(Dense(1, activation = activation_last_layer))
    model_fc.compile(optimizer = optimizer_, loss = loss_ ,metrics=['accuracy'])
#    model_fc.summary()	  
    return (model_fc)