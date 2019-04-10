"""
MLP.py
~~~~~~~~~~
Neural Network Type: Multiple Layers Perceptrons
Cost Error: Cross Entropy
Gradient Descent: Stochastic Gradient Descent
Output Activation: Softmax
Hidden Layer Activation: Sigmoid
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
    
    def feedforward(self, a):
        """feedforward function will return the output of this neural network.
        and a is the input of this network."""
        w_hidden = self.weights[:-1]
        b_hidden = self.biases[:-1]
        if w_hidden == []:
            pass
        else:
            for w, b in zip(w_hidden, b_hidden):
                a = Sigmoid(np.dot(w, a) + b)
        a = Softmax(np.dot(self.weights[-1], a) + self.biases[-1])
        return a

            
    def SGD(self, train_data, train_results, epochs, mini_batch_size, learning_rate, 
            train_moniter_data = None, train_moniter_results = None,
            cv_data = None, cv_results = None, 
            test_data = None, test_results = None):
        """Train this neural network using mini-batch stochastic gradient descent. 
        training_data is list of ndarray. If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.   """
        if test_data: n_test = len(test_data)
        if cv_data: n_cv = len(cv_data)
        n = len(train_data)
        
        for j in range(epochs):
            train_data_shuffled, train_results_shuffled = shuffleTogether(train_data, train_results)
            mini_batches_data = [train_data_shuffled[k:k+mini_batch_size] 
                                 for k in range(0, n, mini_batch_size)]
            mini_batches_results = [train_results_shuffled[k:k+mini_batch_size] 
                                 for k in range(0, n, mini_batch_size)]
            for mini_batch_data, mini_batch_results in zip(mini_batches_data, mini_batches_results): 
                self.update_mini_batch(mini_batch_data, mini_batch_results, learning_rate)
            if test_data:
                print("Epoch {0} Complete with test accuracy: {1:.4f} %".format(j, 100 * self.evaluate(test_data, test_results)/n_test))
            else:
                print("Epoch {0} Complete".format(j))
            if train_moniter_data: print("train accuracy: {0:.4f} %".format(100 * self.evaluate(train_moniter_data, train_moniter_results)/n))
            if cv_data:  print("cv accuracy: {0:.4f} %".format(100 * self.evaluate(train_moniter_data, train_moniter_results)/n_cv))



    def update_mini_batch(self, mini_batch_data, mini_batch_results, learning_rate):
        n = len(mini_batch_data)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in zip(mini_batch_data, mini_batch_results):
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - (learning_rate/n)*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - (learning_rate/n)*nb for b, nb in zip(self.biases, nabla_b)] 


    def backprop(self, x, y):
        """x is trainging example. y is the label.
           backprop will return two lists(nabla_w, nabla_b) of ndarrays each which contains derivative wrt specific weight or bias,
           similar to self.weights and self.biases
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        Activation = x
        Activations = [x]
        zs = []
        ###feedforward part
        w_hidden = self.weights[:-1]
        b_hidden = self.biases[:-1]
        if w_hidden == []:
            pass
        else:
            for w, b in zip(w_hidden, b_hidden):
                z = np.dot(w, Activation) + b
                zs.append(z)
                Activation = Sigmoid(z)
                Activations.append(Activation)
        z = np.dot(self.weights[-1], Activation) + self.biases[-1]
        zs.append(z)
        Activation = Softmax(z)
        Activations.append(Activation)  
        ###backforward part 
        delta = self.cost_derivative(Activation, y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, Activations[-2].transpose())

        for l in range(2,self.num_layers):
            z = zs[-l]
            zp = d_Sigmoid(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * zp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, Activations[-l-1].transpose())
        return (nabla_w, nabla_b)

    def evaluate(self, test_data, test_results):
        tests = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in zip(test_data, test_results)]
        return sum(int(x == y) for (x, y) in tests)


    def cost_derivative(self, output_activations, y):
        return output_activations - y 

        
###helper functions
def Softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def d_Softmax(x):
    return Softmax(x) * (1.0 - Softmax(x))    

def Sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def d_Sigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

def shuffleTogether(A, B):
    if len(A) != len(B):
        raise Exception("Lengths don't match")
    indexes = list(range(len(A)))
    random.shuffle(indexes)
    A_shuffled = [A[i] for i in indexes]    
    B_shuffled = [B[i] for i in indexes]
    return A_shuffled, B_shuffled


