# MLN-Framework
This is a step-by-step multi-layers neural network tutorial. We use python as our language. Two editions will be denoted, **jupyter notebook** and a **.py file**.

## Introduction
### Necessary Packages 
* [numpy](http://www.numpy.org/) is the main package for scientific computing with Python. 
* [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.

## Outline of Framework
### Helper Functions
* sigmoid function 
* derivative of sigmoid function
* relu function
* derivative of sigmoid function

### Initialize Parameters
* input, **python array (list)**, of number of nodes for each layer, for example, a 2 layers network:  
```
layer_dims = [num0, num1, num2]
```
* return **weights** and **bias** for each layers, where wl has shape (layer_dims[l]*layer_dims[l-1]) ,and bl has shape (layer_dims[l],1).   

### Linear Forward Module
#### Linear Forward
The linear forward module (vectorized over all the examples) computes the following equations:

$$Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l-1]}$$

where $A^{[0]} = X$