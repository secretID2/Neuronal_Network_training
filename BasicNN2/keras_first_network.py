# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:52:55 2018

@author: lcristovao
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import pandas as pd
from sklearn import preprocessing

def categoricalToNumeric(array):
    le = preprocessing.LabelEncoder()
    le.fit(array)
    return le.transform(array)
    
    
def TurnDatasetToNumeric(dataset):
        
    for i in range(len(dataset.dtypes)):
        if dataset.dtypes[i]==object:
            v=dataset.iloc[:,i].values
            #print(v)
            v=categoricalToNumeric(v)
            dataset.iloc[:,i]=v
        
    return dataset

#C:\Users\lcristovao\Documents\GitHub\Neuronal_Network_training\BasicNN2

# fix random seed for reproducibility
numpy.random.seed(7)
# Breast cancer dataset
dataset=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
##First columnis patient Id that does not matter
TurnDatasetToNumeric(dataset)
dataset = dataset.iloc[:,1:].values

# split into input (X) and output (Y) variables
X = dataset[:,1:-1]
Y = dataset[:,0]

n_atributes=X.shape[1]#number of columns
# create model
model = Sequential()
model.add(Dense(12, input_dim=n_atributes, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)


# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


print("Input:",X[3,:],"->",Y[3])
prediction=model.predict(X[3:4,:])
print("predict:",round(prediction[0][0]))


#___________Save And Load________________________________________________________
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

'''
# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# Fit the model
loaded_model.fit(X, Y, epochs=150, batch_size=10)


# evaluate the model
scores = loaded_model.evaluate(X, Y)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))
'''
