# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 22:56:24 2018

@author: Lenovo
"""

import keras
import numpy as np
import scipy
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
(train_x_orig, train_y_orig) , (test_x_orig, test_y_orig) = mnist.load_data()

plt.figure(1)
plt.imshow(train_x_orig[2])
plt.show()

np.random.seed(1)
'''
plt.figure(1)
plt.imshow(train_x_orig[10])
plt.show()
print(train_y[10])
'''
### define variables
m_train= train_x_orig.shape[0]
num_px= train_x_orig.shape[1]
m_test= test_x_orig.shape[0]

train_x_flatten= train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten= test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_y_orig= train_y_orig.reshape((60000,1))
test_y_orig= test_y_orig.reshape((10000,1))
train_y= np.zeros((60000, 10))
test_y= np.zeros((10000, 10))
for i in range(60000):
    train_y[i][train_y_orig[i]]=1
for i in range(10000):
    test_y[i][test_y_orig[i]]=1
train_y= train_y.T
test_y= test_y.T
train_x= train_x_flatten/255
test_x= test_x_flatten/255

n_x= train_x.shape[0]
n_h= 30
n_y= train_y.shape[0]

layer_dims= (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h,n_y):
    parameters=[]
    W1= np.random.randn(n_h, n_x)*0.01
    b1= np.zeros((n_h, 1))
    W2= np.random.randn(n_y, n_h)*0.01
    b2= np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def sigmoid(z):
    cache=[]
    result= 1/(1+np.exp(-z))
    cache = (z)
    return result, cache

def relu(z):
    cache= []
    result= np.maximum(0, z)
    cache= (z)
    return result, cache

def linear_activation(A, W, b):
    cache=[]
    Z= np.dot(W, A)+ b
    cache= (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev,W, b, activation):
    cache=[]
    if (activation== 'sigmoid'):
        Z, linear_cache = linear_activation(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif (activation== 'relu'):
        Z, linear_cache = linear_activation(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert(A.shape== (W.shape[0], A_prev.shape[1]))
    cache= (linear_cache, activation_cache)
    return A, cache
 
def L_model_forward(X, parameters):
    caches=[]
   
    A1, cache= linear_activation_forward(X, parameters["W1"], parameters["b1"], activation= 'relu')
    caches.append(cache)
    A2, cache= linear_activation_forward(A1, parameters["W2"], parameters["b2"], activation= 'sigmoid')
    caches.append(cache)
    
    return A2, caches

def compute_cost(A2, Y):
    m= Y.shape[1]
    cost= (-1/m)* np.sum(np.multiply(Y, np.log(A2))+ np.multiply(1-Y, np.log(1-A2)))
    cost= np.squeeze(cost)
    assert(cost.shape== ()) 
    return cost       

def linear_backward(dZ, cache):
    A_prev, W, b= cache
    m= A_prev.shape[0]
    
    dW= (1/m)* np.dot(dZ, A_prev.T)
    db= (1/m)* np.sum(dZ, axis=1, keepdims= True)
    dA_prev= np.dot(W.T, dZ)
    
    assert(dW.shape== W.shape)
    assert(db.shape== b.shape)
    assert (dA_prev.shape == A_prev.shape)
    return dA_prev, dW, db

def sigmoid_backward(dA, cache):
    Z= cache
    s= 1/(1+np.exp(-Z))
    dZ= dA* s*(1-s)
    
    assert(dZ.shape == Z.shape)
    return dZ

def relu_backward(dA, cache):
    Z= cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    
    return dZ
    
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache= cache
    
    if (activation == "relu"):
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
       
    elif(activation == "sigmoid"):
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
        
def L_model_backward(AL, Y, caches):
    grads={}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    dAL= -1*(np.divide(Y, AL)+ np.divide(1-Y, 1-AL))
    current_cache= caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(sigmoid_backward(dAL,current_cache[1]),current_cache[0])

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_backward(sigmoid_backward(dAL, caches[1]), caches[0])
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):   
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
            
    return parameters

def two_layer_model(X, Y, layers_dims, learning_rate = 0.005, num_iterations=3000 , print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = []  
    train_acc=[]
    test_acc=[]                            
    m = X.shape[1]                           
    (n_x, n_h, n_y) = layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0, num_iterations):

        A1, cache1 = linear_activation_forward(X, W1, b1, activation= 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation= 'sigmoid')
       
        cost = compute_cost(A2, Y)
        
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
    
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        
        parameters = update_parameters(parameters, grads, learning_rate= learning_rate)
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        if print_cost and i % 10 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict(X, Y, parameters):
    Y_actual, cache= L_model_forward(X, parameters)
    c=0
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if(Y_actual[i][j]>0.5):
                Y_actual[i][j]=1
            else:
                Y_actual[i][j]=0;
            if(Y_actual[i][j]== Y[i][j]):
                c=c+1
    print(c/(Y.shape[0]* Y.shape[1]))      
            
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)    





