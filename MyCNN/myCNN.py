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
import NN

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





dataset=[]
backdataset=pd.DataFrame([])
matrix_size=10
xpto=[]
Model=NN.NN()
#Model.GetClassifier()

@bt.route('/') # or @route('/login')
def init():
    global backdataset
    names=[i for i in range(matrix_size*matrix_size+1)]
    backdataset=pd.read_csv('myMNIST.txt',names=names)
    #print(backdataset.head)
    return bt.static_file('index.html',root="files/")

@bt.get('/Submit',method='POST')
def Submit():
    global dataset
    Class=bt.request.forms.get('Class')
    Class=np.array([Class])
    
    data=bt.request.forms.get("data")#np.fromstring('\x01\x02', dtype=np.uint8)
    data=[int(i) for i in data.split(',')]#values = [int(i) for i in lineDecoded.split(',')] 
    data=np.array(data)
    data=np.hstack((data,Class))#equivalent to np.concatenate((a,b),axis=1)
    dataset.append(data)
    print(data)
    print(dataset)
    return "OK"

@bt.get('/Save')
def Save():
    global dataset
    global backdataset
    dataset2=np.array(dataset)
    dataset2=pd.DataFrame(dataset2)
    #dataset.to_csv('C:/Users/lcristovao/Documents/GitHub/Neuronal_Network_training/MyCNN/myMNIST.txt',index=None,header=None)
    backdataset=pd.concat([backdataset,dataset2],axis=0)
    #dataset2=TurnDatasetToNumeric(dataset)
    backdataset.to_csv('C:/Users/lcristovao/Documents/GitHub/Neuronal_Network_training/MyCNN/myMNIST.txt',index=None,header=None)
    return 'OK'

@bt.get('/ClassificationPage')
def ClassificationPage():
    global Model
    Model.GetClassifier()
    return bt.static_file('ClassifierPage.html',root="files/")


@bt.get('/Predict',method='POST')
def Predict():
    global Model
    global xpto
    
    data=bt.request.forms.get("data")#np.fromstring('\x01\x02', dtype=np.uint8)
    data=[int(i) for i in data.split(',')]#values = [int(i) for i in lineDecoded.split(',')] 
    #data=np.array(data)
    data=np.array([data])
    data=data.astype('int64')
    xpto.append(data)
    #Model.Predict(xpto)
    return "OK"


bt.run(host='localhost', port=80, server='paste')