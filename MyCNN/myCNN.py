# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:30:42 2018

@author: lcristovao
"""

import bottle as bt
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
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



@bt.route('/') # or @route('/login')
def init():
    return bt.static_file('index.html',root="files/")

bt.run(host='localhost', port=80, server='paste')