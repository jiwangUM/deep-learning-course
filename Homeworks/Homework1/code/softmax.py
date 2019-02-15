import numpy as np
from layers import *


class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights                                  #
        # and biases using the keys 'W' and 'b'                                    #
        ############################################################################
        self.hidden_dim = hidden_dim
#        if hidden_dim is None:
        if not hidden_dim:
            self.params["W1"] = np.random.normal(0, weight_scale, (input_dim,num_classes))
            self.params["b1"] = np.zeros(num_classes)
        else:
            self.params["W1"] = np.random.normal(0, weight_scale, (input_dim,hidden_dim))
            self.params["b1"] = np.zeros(hidden_dim)
            self.params["W2"] = np.random.normal(0, weight_scale, (hidden_dim,num_classes))
            self.params["b2"] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        if not self.hidden_dim:
            a1, cache_a1 = fc_forward(X, self.params["W1"], self.params["b1"])
            scores = a1
        else:
            a1, cache_a1 = fc_forward(X, self.params["W1"], self.params["b1"])
            h1, cache_h1 = relu_forward(a1)
            a2, cache_a2 = fc_forward(h1, self.params["W2"], self.params["b2"])
            scores = a2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_losss(scores, y)
        
        if not self.hidden_dim:
            loss += self.reg/2*(np.linalg.norm(self.params["W1"])**2 + np.linalg.norm(self.params["b1"])**2)
            dx, grads["W1"], grads["b1"] = fc_backward(dout, cache_a1)
            grads["W1"] += self.reg * self.params["W1"]
            grads["b1"] += self.reg * self.params["b1"]
        else:
            loss += self.reg/2*(np.linalg.norm(self.params["W1"])**2 + np.linalg.norm(self.params["b1"])**2 + np.linalg.norm(self.params["W2"])**2 + np.linalg.norm(self.params["b2"])**2)
            dh1, grads["W2"], grads["b2"] = fc_backward(dout, cache_a2)
            da1 = relu_backward(dh1, cache_h1)
            dx, grads["W1"], grads["b1"] = fc_backward(da1, cache_a1)
            grads["W2"] += self.reg * self.params["W2"]
            grads["b2"] += self.reg * self.params["b2"]
            grads["W1"] += self.reg * self.params["W1"]
            grads["b1"] += self.reg * self.params["b1"]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
