# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:03:03 2021

@author: Deepstrats
"""

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convolutional Neural Network : CNN

# CNN : It's a type of Neural network that  is specially designed to work with 2 dimensional data.
# Convolutional Layer : Convolution / Convolve

# Convolution : Linear operation where multiplication of set of weights with input happens

# Filters / Feature Detectors : Neurons of the layer. Weights & o/p of the value

# feature Map : O/p of one filter applied to the previous layer

# Channels : RGB, A filter should have the same number of channels as the input (Depth)

# Zero Padding : Adding an extra column or rows of zeroes to match the input array

# Max pooling : 

# Flattening

# Fully connected layer

# for a gray scale image of 32 w*32 h = 32*32*1 = 1024 pixels

# for a color image of 32 w*32 h = 32*32*3 [RGB]

# Example of a CNN model for a 1-d dataset

'''Load the data

Pre-process the data : Reshape the data into a std. height & width by pixels & convert image into an array
                       Rescale the data 
                       1 channel/depth (if greyscale) otherwise depth = 3'''
                       

data = np.array([0,0,0,1,1,0,0,0]).reshape(1,8,1)
data

type(data)

from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, Conv3D, Dense, Flatten, MaxPooling2D, BatchNormalization

model = Sequential()
model.add(Conv1D(1, 3, input_shape = (8,1)))

f = [np.array([[[0]],[[1]],[[0]]]), np.array([0.0])]
model.set_weights(f)

print(model.get_weights())

res = model.predict(data)

print(res)

# Use a default dataset top build a CNN model

from keras.datasets import mnist, fashion_mnist
m = mnist.load_data()

(x_train,y_train),(x_test,y_test) =  mnist.load_data()

x_train[:5]

# plot a few images
for i in range(9):
    plt.subplot(330+1+i)
    plt.imshow(x_train[i],cmap = plt.get_cmap('gray'))
    plt.show()

x_train = x_train.reshape((x_train.shape[0], 28,28,1))
x_test = x_test.reshape((x_test.shape[0], 28,28,1))

x_test[:10]
x_train[:10]

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

train_norm = x_train.astype('float32')
test_norm = x_test.astype('float32')

train_norm = train_norm / 255.0
test_norm = test_norm / 255.0

# Define the layers of the CNN model

model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', kernel_initializer='glorot_uniform', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
#model.add(Conv2D(64, (3,3), activation = 'relu', kernel_initializer='glorot_uniform', input_shape=(28,28,1)))
#model.add(Conv2D(64, (3,3), activation = 'relu', kernel_initializer='glorot_uniform', input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Dense(10,activation='softmax'))

# Compile the model
# Optional : If you want to set learning rate and momentum manually
from keras.optimizers import SGD
#opt = SGD(lr=0.01, momentum = 0.9)
#model.compile(optimizer= opt,loss = 'categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer="adam",loss = 'categorical_crossentropy',metrics=['accuracy'])
m1 = model.fit(train_norm,y_train, epochs=10, batch_size=25, validation_data=(test_norm, y_test))

_,acc = model.evaluate(test_norm, y_test)
print('%3f' % (acc * 100.0))

'''
for i in range(len(m1.history.keys())):
    plt.subplot(2,1,1)
    plt.title("Cross-entropy loss")
    plt.plot(m1.history['loss'], color = "blue", label ='train')
    plt.plot(m1.history['val_loss'], color = "green", label = "test")
    plt.subplot(2,1,1)
    plt.title("Accuracy")
    plt.plot(m1.history['acc'], color = "blue", label ='train')
    plt.plot(m1.history['val_acc'], color = "green", label = "test")
plt.show()

m1.history.items()

m1.history['val_loss']'''

res = model.predict(test_norm)
res
len(res)

res[0]
import numpy as np
np.round(res[0], 5)

argmax(y_test[0])
y_test[0].argmax()

y_test[:5,:]
res[:5,:]

# Predicting on a new sample

#from tensorflow.keras.utils import load_img
from keras.preprocessing.image import load_img, img_to_array
pwd
img = load_img('nine.png', grayscale = True, target_size = (28,28))
img = img_to_array(img)
img = img.reshape(1,28,28,1)
img = img.astype('float32')
img = img / 255.0

pred_digit = model.predict(img)
np.round(pred_digit[0])

pred_digit.argmax()
pred_digit.max()








