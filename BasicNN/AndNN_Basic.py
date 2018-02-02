# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:07:59 2018

@author: lcristovao
"""

import numpy as np

class Neuron:
    
    @staticmethod 
    def Add(array):
        Sum=0
        for e in array:
            Sum+=e
        return Sum
    
    @staticmethod 
    def Relu(x):
        if x<0:
            return 0
        return x
    
    def __init__(self,weight,bias):
        self.weight=weight
        self.bias=bias
    
    #inputs is an array with input numbers
    def Output(self,inputs):
        out=self.Add(inputs)
        out+=self.bias
        out=self.Relu(out)
        return out
    
    
    def updateWeight(self,error):
        self.weight+=error
    
    
class DenseBrain:
    
    layers=[]
   
    
    
    #layers is an array where its size is the number of brain layers 
    #and the number inside each vector is the number of neurons  per layer
    def __init__(self,nlayers,bias):
        for n_neurons in nlayers:
            neurons=[]
            rn=np.random.random_sample()
            for i in range(n_neurons):
                n=Neuron(rn,bias)
                neurons.append(n)
                
            self.layers.append(neurons)
            
    
    def Propagation(self,inputs):
        
        #shape in each pos has an array with the result of each neuron
        layerout=[]
        #input fase:
        #input neurons
        
        input_neurons=self.layers[0]
        outs=[]
        for neuron in input_neurons:
            result=neuron.Output(inputs)
            out.append(result)
        
        
            
        next_layers=self.layers[1:]
        
        
        for layer in next_layers:
            next_outs=[]
            for neuron in layer:
                for out in outs:
                    result=neuron.Output(out)
        
        return out
        
    def BackPropagation(self,error):
        for l in reversed(self.layers):
            neurons=self.layers[l]
            for neuron in neurons:
                neuron.updateWeight(error/len(neurons))
        
        
    @staticmethod    
    def Error(true_value,output_value):
        return true_value-output_value
        
        
        
        
#____________________Main________________________________        
brain=DenseBrain([2,8,1],0.1)

while True:
    out=brain.Propagation([1,1])
    print("Propagation:",out)
    error=brain.Error(1,out)
    print("error:",error)
    brain.BackPropagation(error*0.1)
        