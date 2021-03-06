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


def SaveToFile(dataset):
        output=""
        for line in dataset:
            for i in range(len(line)):
                if i==len(line)-1:
                    output+=str(line[i])
                else:    
                    output+=str(line[i])+","
            output+="\n"
        with open("myMNIST.txt", "a") as myfile:
            myfile.write(output)


dataset=[]
backdataset=pd.DataFrame([])
matrix_size=10
xpto=[]
normal_data=[]
Model=NN.NN()
Model.GetClassifier()

@bt.route('/') # or @route('/login')
def init():
#    global backdataset
#    names=[i for i in range(matrix_size*matrix_size+1)]
#    backdataset=pd.read_csv('myMNIST.txt',names=names)
#    #print(backdataset.head)
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
#    global backdataset
#    dataset2=np.array(dataset)
#    dataset2=pd.DataFrame(dataset2)
    #dataset.to_csv('C:/Users/lcristovao/Documents/GitHub/Neuronal_Network_training/MyCNN/myMNIST.txt',index=None,header=None)
#    backdataset=pd.concat([backdataset,dataset2],axis=0)
#    #dataset2=TurnDatasetToNumeric(dataset)
#    backdataset.to_csv('C:/Users/lcristovao/Documents/GitHub/Neuronal_Network_training/MyCNN/myMNIST.txt',index=None,header=None)
    SaveToFile(dataset)
    dataset=[]
    return 'OK'

@bt.get('/ClassificationPage')
def ClassificationPage():
#    global Model
#    Model.GetClassifier()
    return bt.static_file('ClassifierPage.html',root="files/")


@bt.get('/Predict',method='POST')
def Predict():
    global Model
    global xpto
    global normal_data
    data=bt.request.forms.get("data")#np.fromstring('\x01\x02', dtype=np.uint8)
    data=[int(i) for i in data.split(',')]#values = [int(i) for i in lineDecoded.split(',')] 
    #data=np.array(data)
    #print(data)
    normal_data.append(data)
    data=np.array([data])
    data=data.astype('int64')
    xpto.append(data.reshape(1,10,10,1))
    Model.Predict(xpto[0])
    return "OK"


bt.run(host='localhost', port=80, server='paste')


#Model.GetClassifier()
#test results
#for number in xpto:
#    print(Model.Predict(number))
##print what was tested    
#s=""
#for number in normal_data:
#    s+="\n"
#    #print(number)
#    number=np.array(number).flatten().reshape(10,10).T
#    for i in range (number.shape[0]):
#        for j in range(number.shape[1]):
#            if(number[i][j]==0):
#                s+=" "
#            else:
#                s+="*"
#        s+="\n"
#print(s)


#Add to DB the tests
#right_answers=[0,1,2,3,4,5,6,7,8,9]
#dataset=[]
#for l in range(len(normal_data)):
#    normal_data[l].append(right_answers[l])
#
#for line in normal_data:
#    dataset.append(np.array(line).flatten())
#    
#SaveToFile(dataset)

######Print training dataset##############################
#dataset=pd.read_csv('myMNIST.txt',sep=",",header=None)
#s=""
#for h in range (dataset.shape[0]):
#    data=dataset.iloc[h,:-1].values.reshape(10,10).T
#    classe=dataset.iloc[h,-1]
#    for i in range (data.shape[0]):
#        for j in range(data.shape[1]):
#            if(data[i][j]==0):
#                s+=" "
#            else:
#                s+="*"
#        s+=str(classe)+"\n"
#        
#        
#with open("DatasetFigures.txt", "a") as myfile:
#    myfile.write(s)