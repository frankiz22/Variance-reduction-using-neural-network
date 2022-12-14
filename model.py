#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
from scipy.linalg import null_space
from tensorflow import keras


# In[2]:


class DNN():
    def __init__(self,N, n = 10, other_layer = 2, neurone_by_layer = 10, with_linear = True, prediction_type = "reg", metric = 'mse'):
        model = keras.Sequential()
        model.add(keras.layers.Input(N))
        if with_linear:
            model.add(keras.layers.Dense(n, activation = keras.activations.linear, use_bias = False))
        
        for i in range(other_layer):
            model.add(keras.layers.Dense(neurone_by_layer, activation = keras.activations.relu,
                                         use_bias=True,bias_initializer=keras.initializers.HeNormal()))
    
            
        if prediction_type == "reg":
            model.add(keras.layers.Dense(1, activation = keras.activations.linear,
                                         use_bias=True,bias_initializer=keras.initializers.HeNormal()))
            model.compile(loss = 'mean_squared_error', 
                        optimizer = keras.optimizers.Adam(0.01),
                        metrics = [metric])
            self.model = model
            
        elif prediction_type == 'bin_class':
            model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid,
                     use_bias=True,bias_initializer=keras.initializers.HeNormal()))
            model.compile(loss = "binary_crossentropy", 
                        optimizer = keras.optimizers.Adam(0.01),
                        metrics = ["binary_accuracy"])

            self.model = model
        else:
            print("name prediction type error. either reg or bin_class")
            return
        
    
    def fit(self, X_train, y_train, eval_set, verbose, min_delta = 0, patience = 10, epochs =1000):

        self.model.fit(X_train, y_train, 
                                batch_size = 32, 
                                epochs = epochs, 
                                validation_data = eval_set,
                                callbacks = keras.callbacks.EarlyStopping(patience = patience, min_delta = min_delta),
                                verbose = verbose)
    def get_matrice_U(self):
        return self.model.get_weights()[0].T
        


# In[3]:


class Extracted_DNN(DNN):
    def __init__(self,model_input):
        model = keras.Sequential()
        n = model_input.layers[0].get_config()["units"]
        model.add(keras.layers.Input(n))
        layers = model_input.layers
        for layer in layers[1:]:
            model.add(layer)
        self.model = model


# In[ ]:




