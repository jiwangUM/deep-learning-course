import numpy as np

from layers import *

class LogisticClassifier(object):
  """
  A logistic regression model with optional hidden layers.
  
  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the model. Weights            #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases (if any) using the keys 'W2' and 'b2'.                        #
    ############################################################################
    self.hidden_dim = hidden_dim
#    if hidden_dim is None:
    if not hidden_dim:
        self.params["W1"] = np.random.normal(0, weight_scale, (input_dim,1))
        self.params["b1"] = np.zeros(1)
    else:
        self.params["W1"] = np.random.normal(0, weight_scale, (input_dim,hidden_dim))
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = np.random.normal(0, weight_scale, (hidden_dim,1))
        self.params["b2"] = np.zeros(1)
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, D)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N,) where scores[i] represents the logit for X[i]
    
    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the model, computing the            #
    # scores for X and storing them in the scores variable.                    #
    ############################################################################
    if not self.hidden_dim:
        a1, cache_a1 = fc_forward(X, self.params["W1"], self.params["b1"])
        scores = a1
    else:
        a1, cache_a1 = fc_forward(X, self.params["W1"], self.params["b1"])
        h1, cache_h1 = relu_forward(a1)
        a2, cache_a2 = fc_forward(h1, self.params["W2"], self.params["b2"])
        scores = a2
       
    #Bug2: If scores is (N,1), need to be (N,) for check_accuracy in solver.py to operate correctly
    #print(scores.shape)
    scores = np.reshape(scores, (scores.shape[0],))
    #print(scores.shape)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the model. Store the loss          #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss and make sure that grads[k] holds the gradients for self.params[k]. #
    # Don't forget to add L2 regularization.                                   #
    #                                                                          #
    ############################################################################
    y = 2*y - 1
    loss, dout = logistic_loss(scores, y)
    
    #Bug1: change (N,) to (N,1) because fc_backward has dot product and need input to be 2d
    dout = np.reshape(dout, (dout.shape[0],1))
    
    #print(dout.shape)
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
