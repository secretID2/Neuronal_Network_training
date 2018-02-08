# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:52:55 2018

@author: lcristovao
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

class NN:
    model=None
    classes=None
    X=None
        
    
    
    
    def GetClassifier(self):
        
        # fix random seed for reproducibility
        np.random.seed(7)
        # Breast cancer dataset
        #dataset=pd.read_csv('myMNIST.txt',sep=",",header=None)
        dataset=pd.read_csv('myMNIST.txt',sep=",",header=None)
        dataset2=dataset
        #Turn columns to numerical
        self.X=dataset2.iloc[:,:-1].values
        self.classes=pd.get_dummies(dataset2.iloc[:,-1])
        Y=self.classes.values
        #NUmber of columns
        number_of_outputs=Y.shape[1]
        ##number of columns
        n_atributes=self.X.shape[1]#number of columns
        #
        
        
        # create model
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=n_atributes, activation='relu'))
        self.model.add(Dense(12, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(number_of_outputs, activation='sigmoid'))
        
        
        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Fit the model
        self.model.fit(self.X, Y, epochs=300, batch_size=10)
        
        
        # evaluate the model
        scores = self.model.evaluate(self.X, Y)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        
        
        #print("Input:",X[0,:],"->",Y[0])
        #prediction=model.predict(X[0:1,:])
        #print("predict:",prediction)
        
        #return model
        #Get top 3 values
    #    print("Input:",X[8,:],"->",classes.columns[Y[8]==1][0])
    #    prediction=model.predict(X[8:9,:])
    #    print("predict:\nClasse: ",classes.columns[prediction.argsort()[0][::-1]][0],"->",prediction[0][prediction.argsort()[0][::-1]][0]*100,"%")
    #    print("predict:\nClasse: ",classes.columns[prediction.argsort()[0][::-1]][1],"->",prediction[0][prediction.argsort()[0][::-1]][1]*100,"%")
    #    print("predict:\nClasse: ",classes.columns[prediction.argsort()[0][::-1]][2],"->",prediction[0][prediction.argsort()[0][::-1]][2]*100,"%")
    
    def Predict(self,data):
            
    #        data2=[data]
            print(data)
            prediction=self.model.predict(data)
            print("predict:\nClasse: ",self.classes.columns[prediction.argsort()[0][::-1]][0],"->",prediction[0][prediction.argsort()[0][::-1]][0]*100,"%")
            print("predict:\nClasse: ",self.classes.columns[prediction.argsort()[0][::-1]][1],"->",prediction[0][prediction.argsort()[0][::-1]][1]*100,"%")
            print("predict:\nClasse: ",self.classes.columns[prediction.argsort()[0][::-1]][2],"->",prediction[0][prediction.argsort()[0][::-1]][2]*100,"%")

print(__name__)

#C=NN()
#C.GetClassifier()
#C.Predict(C.X[0:1])
#GetClassifier()
#Predict(X[0:1])

#___________Save And Load________________________________________________________
# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")

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
