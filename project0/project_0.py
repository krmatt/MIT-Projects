# 6.86x Project 0
# Author: Matthew Kramer

import numpy as np

def neural_network(inputs, weights):
    return np.array([np.tanh(weights[0]*inputs[0] + weights[1]*inputs[1])])

def scalar_function(x, y):
    if x > y:
        return x/y
    else:
        return x*y

def vector_function(x, y):
    vectorized = np.vectorize(scalar_function)
    return vectorized(x, y)

A = np.array([1,5])
B = np.array([2,6])

print(vector_function(A,B))
print(vector_function(B,A))
