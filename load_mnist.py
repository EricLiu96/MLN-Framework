"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data. In practice, ``load_data()`` is the
function usually called by our neural network code.
"""

#Third-party libraries
from scipy.io import loadmat
import numpy as np

def load_data():
    """Return 4 lists that  ``(train_set, train_results, test_set, test_results)`` 
    each train/test set has the format that a list containing 60,000 ndarray entries with dimension (784,1), 
    and each train/test results has the format that a list containing 60,000 ndarray entries with dimension (784,1). 
    """
    x = loadmat('mnist_all.mat')
    train_set = []
    train_set_beta = []
    train_results = []
    test_set = []
    test_set_beta = []
    test_results = []
    for i in range(10):
        train_set_beta.append(x['train' + str(i)])
        train_results += [vectorized_result(i)] * x['train'+str(i)].shape[0]
    train_set_beta = np.concatenate(train_set_beta)
    for i in range(10):
        test_set_beta.append(x['train' + str(i)])
        test_results += [vectorized_result(i)] * x['test'+str(i)].shape[0]
    test_set_beta = np.concatenate(test_set_beta)

    for i in range(train_set_beta.shape[0]):
        temp = np.asarray(train_set_beta[i])
        temp = np.reshape(temp, (train_set_beta.shape[1], 1))
        train_set.append(temp)

    for i in range(test_set_beta.shape[0]):
        temp = np.asarray(test_set_beta[i])
        temp = np.reshape(temp, (test_set_beta.shape[1], 1))
        test_set.append(temp)
    return (train_set, train_results, test_set, test_results)   

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

