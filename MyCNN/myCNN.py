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

import threading

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
        self.done=False

    def GetModel(self):
        self.Model.GetClassifier()
        self.done=True
        
    def Predict(self,data):
        return self.Model.Predict(data)
    
    def DatasetAppend(self,data):
        self.dataset.append(data)
        
    def SaveToFile(self):
        output=""
        for line in self.dataset:
            for i in range(len(line)):
                if i==len(line)-1:
                    output+=str(line[i])
                else:    
                    output+=str(line[i])+","
            output+="\n"
        with open("myMNIST.txt", "a") as myfile:
            myfile.write(output)
        

Id=0
matrix_size=10
clients={}
get_model_threads={}
save_to_file_threads={}
#----------------------

xpto=None


@bt.route('/') # or @route('/login')
def init():
    global Id
    num=0
    #See if user has valid cookie
    for c in clients:
        key = bt.request.get_cookie(clients[c].id)
        #has cookie
        if key!=None:
            if clients[c].password==key:
                #valid user
                return bt.redirect('index.html')
            else:
                return "Not valid user"       
        else:
            num+=1
    #Never was user and is valid
    #Valid user but never enterd site
    if num==len(clients):
        Id+=1
        client=Client(Id)
        clients[Id]=client
        print(Id,client.password)
        bt.response.set_cookie(str(Id), client.password,path='/')
        #bt.response.set_header('Set-Cookie', str(Id)+'='+client.password)
        return bt.redirect('index.html')
    else:
        return "Error! in creating client" 


@bt.route('/index.html')
def InitialPage():
     for c in clients:
        key = bt.request.get_cookie(str(clients[c].id))
        #print(key)
        if key!=None:
            if clients[c].password==key:
                #valid user
                return bt.static_file('index.html',root="files/")
            
    
     return "Not valid user"   
    


@bt.get('/Submit',method='POST')
def Submit():
   for c in clients:
        key = bt.request.get_cookie(str(clients[c].id))
        
        if key!=None:
            if clients[c].password==key:
                #valid user
                client=clients[c]
                Class=bt.request.forms.get('Class')
                Class=np.array([Class])
                print(Class)
                data=bt.request.forms.get("data")#np.fromstring('\x01\x02', dtype=np.uint8)
                data=[int(i) for i in data.split(',')]#values = [int(i) for i in lineDecoded.split(',')] 
                data=np.array(data)
                data=np.hstack((data,Class))#equivalent to np.concatenate((a,b),axis=1)
                client.dataset.append(data)
                return "OK"
            
   return "Fail"


@bt.get('/Save')
def Save():
    for c in clients:
        key = bt.request.get_cookie(str(clients[c].id))
        
        if key!=None:
            if clients[c].password==key:
                #valid user
                client=clients[c]
                t=threading.Thread(target=client.SaveToFile())
                save_to_file_threads[c]=t
                t.start()
                client.dataset=[]
                return 'OK'
        
    return "Fail"

@bt.get('/GetModel')
def GetModel():
    for c in clients:
        key = bt.request.get_cookie(str(clients[c].id))
        
        if key!=None:
            if clients[c].password==key:
                #valid user
                client=clients[c]
                t=threading.Thread(target=client.GetModel())
                get_model_threads[c]=t
                t.start()
                #return bt.static_file("TrainingData.html",root="files/")
                return bt.redirect("ClassificationPage")
                
    return "not valid user!"

@bt.get('/HowItIsGoing')
def Loading():
    for c in clients:
        key = bt.request.get_cookie(str(clients[c].id))
        
        if key!=None:
            if clients[c].password==key:
                #valid user
                client=clients[c]
                if client.done:
                    return "Done"
                else:
                    return "Not Yet"
    
    return "Not Yet"



@bt.get('/ClassificationPage')
def ClassificationPage():
  for c in clients:
        key = bt.request.get_cookie(str(clients[c].id))
        if key!=None:
            if clients[c].password==key:
                #valid user
                return bt.static_file('ClassifierPage.html',root="files/")
                
  return "Not valid user"


@bt.get('/Predict',method='POST')
def Predict():
   for c in clients:
        key = bt.request.get_cookie(str(clients[c].id))
        
        if key!=None:
            if clients[c].password==key:
                #valid user
                client=clients[c]
                data=bt.request.forms.get("data")#np.fromstring('\x01\x02', dtype=np.uint8)
                data=[int(i) for i in data.split(',')]#values = [int(i) for i in lineDecoded.split(',')] 
                #data=np.array(data)
                data=np.array([data])
                data=data.astype('int64')
                client.Predict(data)
                return "OK"
            
   return "Fail"


bt.run(host='localhost', port=80, server='paste')