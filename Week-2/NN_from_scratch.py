import numpy as np
# assume all layers are nx1
# get input as no.of layers, length of each layer
def initialize(n_z_row,n_a_row):
    W= np.random.rand(n_z_row,n_a_row) - 0.5
    B= np.random.rand(n_z_row,1) - 0.5
    return W,B

def ReLU(Z):
    return np.max(Z,0)

def Softmax(X):
    temp= np.sum(np.exp(X))
    return np.exp(X)/temp

def forward_prop():

    return

def grad_ReLU():

    return

def grad_softmax():

    return

def backward_prop():

    return

def update_param():

    return