https://docs.gimp.org/2.8/en/plug-in-convmatrix.html

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 19:06:50 2022

@author: Deepstrats
"""

# Artificial Neural Network

# Installing Theano
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D://data/churn.csv')
x = dataset.iloc[:, 2:12]
y = dataset['Exited']

x1 = pd.get_dummies(x, columns = ['Geography','Gender'], drop_first=True)

# Standardise the data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x1 = sc.fit_transform(x1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x1,y,test_size=0.2, random_state=2)

# Build the layers

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()

# Add the input layers and the hidden layers

model.add(Dense(units = 6, kernel_initializer='glorot_uniform', activation='relu', input_dim = 11))
#model.add(Dropout(rate = 0.1))
model.add(Dense(units=6, kernel_initializer='glorot_uniform', activation = 'relu'))
#model.add(Dropout(rate = 0.1))
model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation = 'sigmoid'))

model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train,y_train, batch_size=25, epochs = 100)

pred = model.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
confusion_matrix(y_test,pred>0.5)
accuracy_score(y_test,pred>0.5)
precision_score(y_test,pred>0.5)

# Weights
# Bias
# Input layers
# Neurons / Learning units
# Hidden layers
# Forward propagation
# Output layer
# Actiavation functions
# Loss fn/cost fn --> Optimzer
# Back-propagation
# Weight adjustments


# Loss fns 

# MSE --> sum(yi-y^)2

# Binary Cross entropy --> -y*(log(y^)-(1-y)*log(1-y^)) 

# Multi class cross entropy

# Activation fns 

# RELU --> max(0,x)
import matplotlib.pyplot as plt
def relu(x):
    return max(0.0,x)

inp = [i for i in range(-10,10)]
op = [relu(i) for i in inp]
plt.plot(inp,op)
plt.show()

# Sigmoid

def sigmoid(x):
    return 1.0/(1.0+exp(-x))

inp = [i for i in range(-10,10)]
op = [sigmoid(i) for i in inp]
plt.plot(inp,op)
plt.show()

# Activation fns

# For hidden layers --> Relu
# for output layer --> Relu for regression, sigmoid for binary classification, softmax for multi class classification

# Cross validation


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def fn(optimizer):
    model= Sequential()
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn= fn,  batch_size = 50, epochs = 10)
accuracies = cross_val_score(estimator = model, X=x_train, y=y_train,cv = 10, n_jobs=1)
mean = accuracies.mean()
variance = accuracies.std()
help(cross_val_score)



# Grid search
from sklearn.model_selection import GridSearchCV
model = KerasClassifier(build_fn = fn)
parameters = {"batch_size" : [25,35],
              "epochs" : [10,20],
              "optimizer" : ["adam","rmsprop"]}

grid_search = GridSearchCV(estimator = model,param_grid = parameters,scoring = "accuracy", cv = 5)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_








