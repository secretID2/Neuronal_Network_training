# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:52:55 2018

@author: lcristovao
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("Indian_diabetes.txt", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,:-1]
Y = dataset[:,-1]

n_atributes=8
# create model
model = Sequential()
model.add(Dense(12, input_dim=n_atributes, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)


# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


print("Input:",X[0,:],"->",Y[0])
print("predict:",round(model.predict(X[0:1,:])))