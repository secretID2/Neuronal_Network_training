# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:50:38 2018

@author: lcristovao
"""

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)



mlp =  MLPClassifier(hidden_layer_sizes=(25,18,16),max_iter=100,random_state=7)
mlp.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test, y_test)))