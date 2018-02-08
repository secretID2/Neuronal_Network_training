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

import random
import string



class Client:
    
    
    def GeneratePass(self):
        self.N=50
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(self.N))
    
    
    def __init__(self,_id):
        #Create obj NN from module NN
        self.Model=NN.NN()
        self.dataset=[]
        self.id=_id
        self.password=self.GeneratePass()

    def GetClassifier(self):
        self.Model.GetClassifier()
        
    def Predict(self,data):
        return self.Model.Predict(data)
    
    def DatasetAppend(self,data):
        self.dataset.append(data)
        
    def SaveToFile(self):
        with open("myMNIST.txt", "a") as myfile:
            myfile.write(self.dataset)
        

Id=0
matrix_size=10
clients={}
get_model_threads={}
#----------------------
dataset=[]
backdataset=pd.DataFrame([])

xpto=None
Model=NN.NN()
Model.GetClassifier()

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
    print(Class)
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
    dataset=[]
    return 'OK'

@bt.get('/ClassificationPage')
def ClassificationPage():
   
    return bt.static_file('ClassifierPage.html',root="files/")


@bt.get('/Predict',method='POST')
def Predict():
    global Model
    global xpto
    data=bt.request.forms.get("data")#np.fromstring('\x01\x02', dtype=np.uint8)
    data=[int(i) for i in data.split(',')]#values = [int(i) for i in lineDecoded.split(',')] 
    #data=np.array(data)
    xpto=np.array([data])
    xpto=xpto.astype('int64')
    
    Model.Predict(xpto)
    return "OK"


bt.run(host='localhost', port=80, server='paste')