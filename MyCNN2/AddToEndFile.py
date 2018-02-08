# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:57:59 2018

@author: lcristovao
"""
import threading

def f(s):
    with open("test.txt", "a") as myfile:
        myfile.write(s)
    
t1=threading.Thread(target=f("111\n")) 
                                                                            
t2=threading.Thread(target=f("222\n"))
t3=threading.Thread(target=f("333\n"))



t1.start()
t2.start()
t3.start()