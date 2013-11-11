
from numpy import *

def perceptron(weights, X, y, alpha, numIter):
    '''Return 1 if weights(i,j) * X(i) > 0 and 0 otherwise and update weights to minimize cost'''
    for i in range(numIter):
        g = h(weights,X)                    # Simple X.w operation 
        p = where(g > 0, 1, 0)              # Give an array 

        weights -= alpha * (X.T.dot(p - y)) # Update weights
    return weights

def h(weights, X):
    '''Return X.dot(weights)'''
    return X.dot(weights)
