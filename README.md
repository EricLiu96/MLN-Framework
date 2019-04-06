# MLP-Framework
This is a step-by-step multi-layers perceptrons neural network tutorial. We use python as our language.
## Introduction
### Necessary Packages 
* [numpy](http://www.numpy.org/) is the main package for scientific computing with Python. 
* [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.

## Outline of Framework
We hope to build a **MLP** class to denote our network. Following 
### Helper Functions

* relu function
* derivative of relu function

### Initialize Parameters
* input, **python array (list)**, of number of nodes for each layer, for example, a 2 layers network:  
```
layer_dims = [num0, num1, num2]
```
* set **weights** and **bias** for each layers, where $w_{l}$ has shape (layer_dims[l] * layer_dims[l-1]) ,and $b_{l}$ has shape (layer_dims[l],1).   

### Linear Forward Module
#### Linear activation Forward
This function will compute out the result of our neural network. The linear forward module (vectorized over all the examples) computes the following equations:

$$Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l-1]}$$

where $A^{[0]} = X$. After linear forward part, we should take it through activation function. There will be 2 kinds of activation functions.


* ReLu: $A = ReLu(Z) = max(0, Z)$
* [Softmax](https://en.wikipedia.org/wiki/Softmax_function): $A = softmax(Z) = \frac{e^{Z_{j}}}{\sum_{k=1}^{K}e^{Z_{k}}}$ 

After iterating these two parts $L$ times, we can get the final result of our neural network.

* input **X, self**
* return **A^{L}**

### [Cross Entropy Error Function](https://sefiks.com/2017/12/17/a-gentle-introduction-to-cross-entropy-loss-function/)

We need to know the derivative of loss function to back propagate. Our cost function is multi-class cross entropy function. Its defintion follows, 

$$E = -\sum_{i=\#class}c_{i}log(A^{L}_{i})$$ 

Where $c_{i}$ is the lable for class $i$, and $A^{L}_{i}$ is the prediction of probability belonging to class $i$.

Notice that we would apply softmax to calculated neural network scores($Z^{L}$) and predict out probabilities first. Cross entropy is applied to softmax applied probabilities and one hot encoded classes calculated second. That’s why, we need to calculate the derivative of total error with respect to the each score.  

![chain-rule-for-cross-entropy1](https://i1.wp.com/sefiks.com/wp-content/uploads/2017/12/chain-rule-for-cross-entrophy-v11.png?zoom=2&resize=665%2C435&ssl=1)

We apply chain rule to calculate the derivative.

![chain-rule-for-cross-entropy1](https://i1.wp.com/sefiks.com/wp-content/uploads/2017/12/chain-rule-for-cross-entrophy-v21.png?zoom=2&resize=665%2C458&ssl=1)

Calculating it step by step, for a specific score with index $i$, 
$$\frac{\partial E}{\partial Z^{L}_{i}} = \sum_{j}(\frac{\partial E}{\partial A^{L}_{j}})(\frac{\partial A^{L}_{j}}{\partial Z^{L}_{i}}) = (\frac{\partial E}{\partial A^{L}_{i}})(\frac{\partial A^{L}_{i}}{\partial Z^{L}_{i}})$$

Considering about $\frac{\partial E}{\partial A^{L}_{i}}$ first, 

$$
\begin{align*}
\frac{\partial E}{\partial A^{L}_{i}} &= \frac{\partial(E = -\sum_{i=\#class}c_{i}log(A^{L}_{i}))}{\partial A^{L}_{i}}\\
&=\frac{\partial(-c_{i}log(A^{L}_{i}))}{\partial A^{L}_{i}}\\
&=-\frac{c_{i}}{A^{L}_{i}}
\end{align*}
$$
From [here] (https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/) we can know that the partial derivative of $\frac{\partial A^{L}_{i}}{\partial Z^{L}_{i}}$ is that,

$$\frac{\partial A^{L}_{i}}{\partial Z^{L}_{i}} = A^{L}_{i}(1 - A^{L}_{i})$$

Thus, we now can calculate out $\frac{\partial E}{\partial Z^{L}_{i}}$, 

$$
\begin{align*}
\frac{\partial E}{\partial Z^{L}_{i}} &= (\frac{\partial E}{\partial A^{L}_{i}})(\frac{\partial A^{L}_{i}}{\partial Z^{L}_{i}})\\
&=-\frac{c_{i} A^{L}_{i}(1 - A^{L}_{i})}{A^{L}_{i}}\\
&=-c_{i}(1 - A^{L}_{i})
\end{align*}
$$
$$\frac{\partial E}{\partial Z^{L}} = -c \odot(1 - A^{L})$$

* input **$A^{L}$, c**

* output **$\frac{\partial E}{\partial Z^{L}}$**

### Backpropagation Algorithm
If you want to look at the detailed explanation of , please click [here](http://neuralnetworksanddeeplearning.com/chap2.html#proof_of_the_four_fundamental_equations_(optional)). If you are not such crazy about mathematics. Here follows a concise, but convincible conduction of backpropagation algorithm.

Backpropagation is about understanding how changing the weights and biases in a network changes the cost function. Ultimately, this means computing the partial derivatives $\frac{\partial C}{\partial W_{jk}^{l}}$ and $\frac{\partial C}{\partial b_{j}^{l}}$, where $W_{jk}^{l}$ denotes the weight connecting between $k_{th}$ neuron in the $l-1_{th}$ layer and $j_{th}$ in the $l_{th}$ layer.

Having a revisiting of we have said in Linear activation forward, for every layer $l = 1,...,L$,

$$A^{l} = \sigma (W^{l}A^{l-1} + b^{l})$$
where $\sigma$ is the activation function, it can be $softmax$ or $relu$. 

Since we aleady have output error,
$$\frac{\partial E}{\partial Z^{L}_{i}} = -c_{i}(1 - A^{L}_{i})$$
$$\frac{\partial E}{\partial Z^{L}} = -c \odot(1 - A^{L})$$
where $\odot$ is element wise multiply.

Now considering about the error $\frac{\partial E}{\partial Z^{l}}$ in terms of the error in the next layer, $\frac{\partial E}{\partial Z^{l+1}}$. We can do this using the chain rule,
$$
\frac{\partial E}{\partial Z_{j}^{l}} = \sum_{k} \frac{\partial E}{\partial Z_{k}^{l+1}} \frac{\partial Z_{k}^{l+1}}{\partial Z_{j}^{l}}
$$ 

To evaluate the first term on the last line, note that,

$$Z_{k}^{l+1} = \sum_{j}W_{kj}^{l+1}A_{j}^{l} + b_{k}^{l+1} = \sum_{j}W_{kj}^{l+1}\sigma(Z_{j}^{l}) + b_{k}^{l+1}$$
Differentiating, we obtain,
$$\frac{\partial Z_{k}^{l+1}}{\partial Z_{j}^{l}} = W_{kj}^{l+1} \sigma^{\prime}(Z_{j}^{l})$$
Substituting back,
$$
\begin{align*}\frac{\partial E}{\partial Z_{j}^{l}} &= \sum_{k}W_{kj}^{l+1} \frac{\partial E}{\partial Z_{k}^{l+1}} \sigma^{\prime}(Z_{j}^{l})\\
\frac{\partial E}{\partial Z^{l}}&= ((W^{l+1})^{T})\frac{\partial E}{\partial Z_{k}^{l+1}}) \odot \sigma^{\prime}(Z^{l})
\end{align*}
$$
Next part we will calculate $\frac{\partial C}{\partial W_{jk}^{l}}$ and $\frac{\partial C}{\partial b_{j}^{l}}$,
$$\begin{align*}
\frac{\partial E}{\partial b_{j}^{l}} &= \sum_{k} \frac{\partial E}{\partial Z_{k}^{l}} \frac{\partial Z_{k}^{l}}{\partial b_{j}^{l}} =\frac{\partial E}{\partial Z_{j}^{l}}\\
\frac{\partial E}{\partial b^{l}}&= \frac{\partial E}{\partial Z^{l}}\\
\\
\frac{\partial E}{\partial W_{jk}^{l}} &= \sum_{i}\frac{\partial E}{\partial Z_{i}^{l}}\frac{\partial Z_{i}^{l}}{\partial W_{jk}^{l}} = \frac{\partial E}{\partial Z_{j}^{l}}A_{k}^{l-1}\\
\frac{\partial E}{\partial W^{l}} &=  \frac{\partial E}{\partial Z^{l}}(A^{l-1})^{T}
\end{align*}$$ 

* input **self, X, c**
* ouput **$\frac{\partial E}{\partial b^{l}}$,$\frac{\partial E}{\partial W^{l}}$ from l =1,...,L**

---
**Summary: the equations of backpropagation**

* $\frac{\partial E}{\partial Z^{L}} = -c \odot(1 - A^{L})$

* $\frac{\partial E}{\partial Z^{l}} = ((W^{l+1})^{T})\frac{\partial E}{\partial Z_{k}^{l+1}}) \odot \sigma^{\prime}(Z^{l})$
* $\frac{\partial E}{\partial b^{l}} = \frac{\partial E}{\partial Z^{l}}$
* $\frac{\partial E}{\partial W^{l}} =  \frac{\partial E}{\partial Z^{l}}(A^{l-1})^{T}$


---
Right here, we omit details of Stochastic Gradient Descent and update function temporarily.