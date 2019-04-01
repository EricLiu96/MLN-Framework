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

<math display="block">
<math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>Z</mi><mfenced open="[" close="]"><mi>l</mi></mfenced></msup><mo>=</mo><msup><mi>W</mi><mfenced open="[" close="]"><mi>l</mi></mfenced></msup><msup><mi>A</mi><mfenced open="[" close="]"><mrow><mi>l</mi><mo>-</mo><mn>1</mn></mrow></mfenced></msup><mo>+</mo><msup><mi>b</mi><mfenced open="[" close="]"><mi>l</mi></mfenced></msup></math>
</math>

where <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>A</mi><mrow><mo>[</mo><mn>0</mn><mo>]</mo></mrow></msup><mo>=</mo><mi>X</mi></math>.


















