# MLP-Framework
This is a step-by-step multi-layers perceptrons neural network tutorial. We use python as our language.
## Introduction
### Necessary Packages 
* [numpy](http://www.numpy.org/) is the main package for scientific computing with Python. 
* [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
* [scipy](https://www.scipy.org/) is the SciPy package of key algorithms and functions core to Python's scientific computing capabilities. 

### Tutorial for load_mnist.py
loading...

## Outline of Framework
We hope to build a **MLP** class to denote our network. Following 
### Helper Functions

* relu function
* derivative of relu function
* softmax function
* derivative of softmax function

### Initialize Parameters
* input, **python array (list)**, of number of nodes for each layer, for example, a 2 layers network:  
```
layer_dims = [num0, num1, num2]
```
* set **weights** and **bias** for each layers, where wl has shape (layer_dims[l] * layer_dims[l-1]) ,and bl has shape (layer_dims[l],1).   

### Linear Forward Module
This function will compute out the result of our neural network. 

After iterating linear forward module and activation **L** times, we can get the final result of our neural network.

* input **X, self**
* return **AL**

### [Cross Entropy Error Function](https://sefiks.com/2017/12/17/a-gentle-introduction-to-cross-entropy-loss-function/)

* input **AL, c**

* output **dZL**

### Backpropagation Algorithm
If you want to look at the detailed explanation of , please click [here](http://neuralnetworksanddeeplearning.com/chap2.html#proof_of_the_four_fundamental_equations_(optional)). If you are not such crazy about mathematics. **Helper** contains a concise, but convincible conduction of backpropagation algorithm.

* input **self, X, c**
* ouput **dbl, dWl from l =1,...,L**

### Mini-bach Stochastic Gradient Descent

### Update Function


