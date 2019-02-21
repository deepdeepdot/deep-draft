Deep Learning
----

### A. Go over our gradient descent solution and error loss function

General solution + concepts

Backpropagation? using a graph example!?
Advantage of a computational graph: we can compute the gradient at each step using chain rule and back-propagation

Hence, all these ML frameworks are based on graph!?

Last time    Y = lambda x: x**2 - 3*x + 10

     Y = lambda x: x**2 - 3*x + 10
     x = np.linspace(-5,10,100)
     plt.plot(x, Y(x))

     start = (10, Y(10))

Chain Rule of slopes

x=2 -> Y(x) -> can determine the gradients backwards

x=2 -> Y(x) = 10, Y(x) = -2 (when skipping +10)

Skipping +10 operation?
    x
       ** (exp)  A(x) = x**2 = 4
    2
                            C(x) = A(x) - B(x) = -2
    3
       *         B(x) = 3 * x =  6
    x

gradient at C: 1

dC/dC = 1
dC/dA = 1  -> for every A units, we change those in C
    gradient at A: 1

dC/dB = -1
    gradient at B: -1

I know I can code this!!!
At least for a basic back-propagation with the computational nodes we have!

Y = lambda x: x**2 - 3*x + 10

Addition (at least for backprop)
+constant !=  A + B computational nodes

Support for a simple backpropagation framework based on
polynomials on x
    +nodes
    -nodes
    +constant
    *constant
    **exponent

Given an input -> forward... then backprop to get gradients
due to the chain rule: just multiply ;)

-> Just for fun and as personal exercise


-----


    (x-5)(x+3)  = x**2 - 2*x - 15
    df/dx = 2x - 2


B. Go over the 11 lines for binary output: single layer
C. FeedForward (FC) continue with 2 layers (fully connected)
D. Transform the custom gradient descent using neural networks!

E. Go over MNIST classifier -> softmax
F. Summary of concepts

G. Curve fitting with tensorflow api?

H. Zoo of neurons architecture


-> Coding practice? Image classifier for:


### The brain


### The perceptron model

Activation neuron: sigmoid function()

sigmoid(x) = (1/1 + exp(-x))


### Activation functions

* Sigmoid Activation
    sigmoid(x) = (1/1+exp(-x))

* Hyperbolic Tangent Activation 
    tanh(x) = sinh(x) / cosh(x)
            = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
            = (exp(2*x)-1) / (exp(2*x)+1)

* Rectified Linear Activation
    relu(x) = max(0, x)

    - Tends to give neurons with sparse activities -> Lots of zeros


### NN using 11 lines?

Machine Learning in 11 lines - Alexander Trask
https://iamtrask.github.io/2015/07/12/basic-python-network/
https://iamtrask.github.io/2015/07/27/python-network-part2/

        To predict
        Inputs	        Output
        0	0	1	    0
        1	1	1	    1
        1	0	1	    1
        0	1	1	    0


import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in range(50000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

    if iter % 5000 == 0:
        print({ "Error": l1_error, "delta": l1_delta, "weights": syn0 })

print("Output After Training:", l1)
print("Weights:", syn0)

nonlin(np.dot(syn0.T, [0, 0, 1]))
nonlin(np.dot(syn0.T, [1, 1, 1]))



### 3-layer Neural Network

import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
            [1],
            [1],
            [0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
hidden_nodes = 4
syn0 = 2*np.random.random((3,hidden_nodes)) - 1
syn1 = 2*np.random.random((hidden_nodes,1)) - 1

for j in range(60000):
    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2
    
    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


How to make a prediction given weights l1, l2?

x = np.array([0, 0, 1])
l1 = nonlin(np.dot(x, syn0))
l2 = nonlin(np.dot(l1,syn1))

x = np.array([1, 1, 1])
l1 = nonlin(np.dot(x, syn0))
l2 = nonlin(np.dot(l1,syn1))

x = np.array([0, 1, 1])
l1 = nonlin(np.dot(x, syn0))
l2 = nonlin(np.dot(l1,syn1))



https://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html


# 2-layer Neural Networks in 11 lines

        X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
        y = np.array([[0,1,1,0]]).T
        syn0 = 2*np.random.random((3,4)) - 1
        syn1 = 2*np.random.random((4,1)) - 1
        for j in range(60000):
            l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
            l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
            l2_delta = (y - l2)*(l2*(1-l2))
            l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
            syn1 += l1.T.dot(l2_delta)
            syn0 += X.T.dot(l1_delta)


### Loss functions

    Prediction: Continuous/numerical -> Regression loss
    Category:
        -Binary. Cross entropy loss
        -Multi-class. Softmax log loss


### Image classifier using MNIST

MNIST
https://js.tensorflow.org/tutorials/mnist.html


----

Deep Learning?
-> Neural Networks architectures

http://www.asimovinstitute.org/neural-network-zoo/





### Computational Graph and Backpropagation
  Why do we need a computational graph?
  Answer: backpropagation

  TODO: Keep working on the Intuition

  while True:
      data_batch = dataset.sample_data_batch()
      loss = graph.forward(data_batch)
      dx = graph.backward() # gradient
      x += learning_rate * dx
