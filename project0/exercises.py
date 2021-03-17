import numpy as np

def randomization(n):
    return np.random.random([n,1])

def operations(h, w):
    A = np.random.random([h,w])
    B = np.random.random([h,w])
    s = A + B

    return A, B, s

def norm(A, B):
    matsum = A + B
    s = np.linalg.norm(matsum)

    return s
