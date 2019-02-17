import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, dropout = False, batch_norm = False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    - dropout: True turn on the dropout
    - batch_norm: True turn on the batch_norm
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################    
    self.batch_norm = batch_norm
    self.dropout = dropout
    
    C, H, W = input_dim
    pool_size = 2
    stride = 2
    
    H_prime = H - (filter_size-1) 
    W_prime = W - (filter_size-1)
    H_pool = 1 + (H_prime - pool_size) // stride 
    W_pool = 1 + (W_prime - pool_size) // stride
    
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size)).astype(dtype)
    #self.params['b1'] = np.zeros((num_filters, H_prime, W_prime)).astype(dtype)
    self.params['b1'] = np.zeros(num_filters).astype(dtype)
    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters*H_pool*W_pool, hidden_dim)).astype(dtype)
    self.params['b2'] = np.zeros(hidden_dim).astype(dtype)
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes)).astype(dtype)
    self.params['b3'] = np.zeros(num_classes).astype(dtype)
    
    self.dropout_param = {
            'p': 0.5,
            'mode': 'train'
            }
    self.bn_param = { 
            'mode': 'train',
            'eps': 1e-5,
            'momentum': 0.9
            }
    
    if self.batch_norm:
        self.params['gamma'] = np.random.normal(0, weight_scale, (num_filters * H_prime * W_prime)).astype(dtype)
        self.params['beta']  = np.zeros(num_filters * H_prime * W_prime).astype(dtype)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
#    for k, v in self.params.iteritems():
#      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if y is None:
        self.dropout_param['mode'] = 'test'
        self.bn_param['mode'] = 'test'
    else:
        self.dropout_param['mode'] = 'train'
        self.bn_param['mode'] = 'train'
    
    
    a1, cache_a1 = conv_forward(X, W1)
    a1 = a1 + b1.reshape(1,b1.shape[0],1,1)
    
    if self.batch_norm:
        a1_bn, cache_a1_bn = batchnorm_forward(a1.reshape((a1.shape[0],-1)), self.params['gamma'], self.params['beta'], self.bn_param)
        a1 = a1_bn.reshape(a1.shape)
        
    h1, cache_h1 = relu_forward(a1)
    h1_maxpool, cache_h1_maxpool = max_pool_forward(h1, pool_param)
    a2, cache_a2 = fc_forward(h1_maxpool, W2, b2)
    h2, cache_h2 = relu_forward(a2)
    
    if self.dropout:
        h2_dropout, cache_h2_dropout = dropout_forward(h2, self.dropout_param)
        h2 = h2_dropout / self.dropout_param['p']
    
    a3, cache_a3 = fc_forward(h2, W3, b3)
    scores = a3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    loss += self.reg/2*(np.linalg.norm(self.params["W1"])**2 
                        + np.linalg.norm(self.params["b1"])**2 
                        + np.linalg.norm(self.params["W2"])**2 
                        + np.linalg.norm(self.params["b2"])**2
                        + np.linalg.norm(self.params["W3"])**2 
                        + np.linalg.norm(self.params["b3"])**2)
    
    dh2, grads['W3'], grads['b3'] = fc_backward(dout, cache_a3)
    
    if self.dropout:
        dh2 = dh2 / self.dropout_param['p']
        dh2 = dropout_backward(dh2, cache_h2_dropout)
        
    da2 = relu_backward(dh2, cache_h2)
    dh1_maxpool, grads['W2'], grads['b2'] = fc_backward(da2, cache_a2)
    dh1 = max_pool_backward(dh1_maxpool, cache_h1_maxpool)
    da1 = relu_backward(dh1, cache_h1)
    
    if self.batch_norm:
        da1_bn, grads['gamma'], grads['beta'] = batchnorm_backward(da1.reshape((da1.shape[0],-1)), cache_a1_bn)
        da1 = da1_bn.reshape(da1.shape)
    
    #grads['b1'] = da1.sum(axis=0)
    grads['b1'] = da1.sum(axis=0).sum(axis=1).sum(axis=1)
    dx, grads['W1'] = conv_backward(da1, cache_a1)
    
    #print(grads['b1'].shape)
    #print(b1.shape)
    grads['W1'] += self.reg * self.params['W1']
    grads['b1'] += self.reg * self.params['b1']
    grads['W2'] += self.reg * self.params['W2']
    grads['b2'] += self.reg * self.params['b2']
    grads['W3'] += self.reg * self.params['W3']
    grads['b3'] += self.reg * self.params['b3']
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
