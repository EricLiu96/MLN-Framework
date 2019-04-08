"""
MLP.py
~~~~~~~~~~
Neural Network Type: Multiple Layers Perceptrons
Cost Error: Cross Entropy
Gradient Descent: Stochastic Gradient Descent
Output Activation: Softmax
Hidden Layer Activation: Relu
"""


#### Libraries 
# Standard library
import random

# Third-party library
import numpy as np

class MLP(object):

    def __init__(self, sizes):
        """sizes should be a list containing number of perceptron for all latyers. 
        For example, sizes can be assigned as [2, 3, 1], which means the input layer has 2 neurons,
        the only hidden layer has 3 neurons, and output layer has 1 neuron.
        The biases and weights for the network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.
        """
        self.num_layers  = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feeforward(self, a):
        # do something interesting
        return a
    def SGD(self, training_data, training_results, epochs, mini_batch_size, learning_rate, test_data = None):
        #do something interesting here
    
    def update_mini_batch(self, mini_batch, learning_rate):
        #do something interesting here

    def backprop(self, x, y):
        """x is trainging example. y is the label.
           backprop will return two lists(nabla_b, nabla_w) of ndarrays each which contains derivative wrt specific weight or bias,
           similar to self.weights and self.biases
           """
        #do something interesting here
    
    def evaluate(self, test_data, test_result):
        #do something interesting here

    def cost_derivative(self, output_activations, y):
        #do something interesting here

###helper functions
    def sigmoid(z):
        #do something interesting here

    def d_sigmoid(z):
        #do something interesting here

    def relu(z):
        #do something interesting here

    def d_relu(z):
        #do something interesting here        




