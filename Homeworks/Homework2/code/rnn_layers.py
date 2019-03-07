"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""
import numpy as np

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """

    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
    cache = (x, prev_h, Wx, Wh, b)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass
    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """

    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, prev_h, Wx, Wh, b = cache
    z = x.dot(Wx) + prev_h.dot(Wh) + b
    #dnext_h: (N, H)
    dz = 1 - np.tanh(z)**2 #(N, H) derivative of tanh
    dz = dnext_h * dz #(N, H)
    
    dx = dz.dot(Wx.T) #****Important: draw the feedward fully-connect layer, Y = x.dot(W) <-> dx = Y.dot(W) [reverse]
    dprev_h = dz.dot(Wh.T)
    dWx = x.T.dot(dz) #****Important: accumulation along N dimension 
    dWh = prev_h.T.dot(dz) #****Important
    db = dz.sum(axis=0)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.
    Inputs:
    - x: Input data for the entire timeseries, of shape (T, N, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (T, N, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    T, N, D = x.shape
    N, H = h0.shape
    h = np.empty((T, N, H))
    h_cache = np.empty((T, N, H))
    prev_h = h0
    for t in range(x.shape[0]):
        h_cache[t] = prev_h
        next_h, _ = rnn_step_forward(x[t], prev_h, Wx, Wh, b)
        h[t] = next_h
        prev_h = next_h
    
    cache = (x, h_cache, Wx, Wh, b)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (T, N, H)
    Returns a tuple of:
    - dx: Gradient of inputs, of shape (T, N, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    x, prev_h, Wx, Wh, b = cache
    T = dh.shape[0]
    
    #dx = np.empty(x.shape)
    dx = np.empty_like(x)
    dWx = np.zeros(Wx.shape)
    dWh = np.zeros(Wh.shape)
    db= np.zeros(b.shape)
    
    dL_t = np.zeros_like(dh[0])
    for t in range(T-1, -1, -1):
        cache_step = (x[t], prev_h[t], Wx, Wh, b)
        dx[t], dL_t, dWx_t, dWh_t, db_t = rnn_step_backward(dh[t] + dL_t, cache_step)
        dWx += dWx_t
        dWh += dWh_t
        db  += db_t
        
    dh0 = dL_t
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    cache = None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    # For Wx of shape D x 4H, you may assume they are the sequence of parameters#
    # for forget gate, input gate, concurrent input, output gate. Wh and b also #
    # follow the same order.                                                    #
    #############################################################################
    N, H = prev_h.shape
    A = x.dot(Wx) + prev_h.dot(Wh) + b #(N, 4H)
    f = sigmoid(A[:, :H]) #(N, H)
    i = sigmoid(A[:, H:2*H])
    c = np.tanh(A[:, 2*H:3*H])
    o = sigmoid(A[:, 3*H:])
    
    next_c = f * prev_c + i * c
    next_h = o * np.tanh(next_c)
    
    cache = (x, prev_h, prev_c, Wx, Wh, b, f, i, c, o, next_c)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db, dprev_h, dprev_c = None, None, None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, prev_h, prev_c, Wx, Wh, b, f, i, c, o, next_c = cache
    N, H = prev_h.shape
    
    do_t = dnext_h * np.tanh(next_c)
    dnext_c = dnext_c + dnext_h * o * (1 - np.tanh(next_c)**2)
    
    df_t = dnext_c * prev_c
    di_t = dnext_c * c
    dc_t = dnext_c * i
    
    dprev_c = dnext_c * f
    dAf = df_t * f * (1-f)
    dAi = di_t * i * (1-i)
    dAc = dc_t * (1-np.power(c,2))
    dAo = do_t * o * (1-o)
    
    dA = np.hstack((dAf, dAi, dAc, dAo))
    
    dx = dA.dot(Wx.T)
    dprev_h = dA.dot(Wh.T)
    dWx = x.T.dot(dA)
    dWh = prev_h.T.dot(dA)
    db = dA.sum(axis=0)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.
    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
    Inputs:
    - x: Input data of shape (T, N, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (T, N, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    T, N, D = x.shape
    N, H = h0.shape
    c = np.zeros_like(h0)
    
    h = np.empty((T, N, H))
    next_h = h0
    next_c = c
    cache = []
    for t in range(T):
        next_h, next_c, cache_t = lstm_step_forward(x[t], next_h, next_c, Wx, Wh, b)
        h[t] = next_h
        cache.append(cache_t)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (T, N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (T, N, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    T, N, H = dh.shape
    x, prev_h, prev_c, Wx, Wh, b, f, i, c, o, next_c = cache[0]
    N, D = x.shape

    dprev_h = np.zeros_like(dh[0])
    dprev_c = np.zeros_like(dh[0])
    dx = np.empty((T,N,D))
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    
    for t in range(T-1, -1, -1):
        dx[t], dprev_h, dprev_c, dWx_t, dWh_t, db_t = lstm_step_backward(dh[t] + dprev_h, dprev_c, cache[t])
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
        
    dh0 = dprev_h

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.
    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    out = W[x, :]
    cache = x, W
    return out, cache

def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    HINT: Look up the function np.add.at
    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass
    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW



def temporal_fc_forward(x, w, b):
    """
    Forward pass for a temporal fully-connected layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.
    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)
    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    D, M = w.shape
    x2d = x.reshape((-1, D))
    out = x2d.dot(w) + b
    out = out.reshape((N, T, M))
    
    cache = (x, w, b)
    return out, cache


def temporal_fc_backward(dout, cache):
    """
    Backward pass for temporal fully-connected layer.
    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass
    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b = cache
    N, T, D = x.shape
    D, M = w.shape
    dout2d = dout.reshape((-1, M))
    x2d = x.reshape((-1, D))
    
    dx = dout2d.dot(w.T).reshape((N, T, D))
    dw = x2d.T.dot(dout2d)
    db = dout2d.sum(axis=0)
    
    return dx, dw, db



def temporal_softmax_loss(x, y, mask):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.
    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.
    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.`
    """
    
    N, T, V = x.shape
#    x2d = x.reshape((N*T, V))
    x2d = x.reshape(N*T, V)
    mask1d = (mask.reshape(N*T) == 1)
    y1d_mask = y.reshape(N*T)[mask1d]
    
#    [BUG1]: Mask is not boolean array but int array
#    x2d_exp_mask = x2d[mask1d] (dim: (60, 8)) vs x2d_exp_mask = x2d[mask1d == 1] (dim: (26, 8))
#    x2d_exp_mask = np.exp(x2d[mask1d]) 
    x2d_mask = x2d[mask1d]
    x2d_mask_exp = np.exp(x2d_mask - np.max(x2d_mask, axis=1, keepdims=True)) #substract max can prevent exp(a really big number)
    x2d_mask_exp_sum = np.sum(x2d_mask_exp, axis=1, keepdims=True)
    x2d_mask_softmax = x2d_mask_exp / x2d_mask_exp_sum
    
    #[BUG2]: 2D array index into 1D has to do things like this:
    #np.arange() is like range() in python
#    loss = -np.sum(np.log(x2d_mask_exp[np.arange(y1d_mask.shape[0]), y1d_mask] / x2d_mask_exp_sum)) / N
#    Not like this:
#    loss = np.sum(-np.log(x2d_mask_exp[y_mask] / x2d_mask_exp_sum)) / N

    #BUG3: Dimention Problem of 1D !!!!
#    debug0 = x2d_mask_exp[np.arange(y1d_mask.shape[0]), y1d_mask] --> this will generate (26,) array
#    debug1 = x2d_mask_exp[np.arange(y1d_mask.shape[0]), y1d_mask] / x2d_mask_exp_sum --> (26,)/(26,1) is not (26,) but (26,26)!!!

    loss = -np.sum(np.log(x2d_mask_softmax[np.arange(y1d_mask.shape[0]), y1d_mask])) / N
 
    dx2d = np.zeros_like(x2d)
    dx2d[mask1d] = x2d_mask_softmax

    #[BUG4]: Slicing/view doesn't work this way
#    dx2d[mask1d][y1d_mask] -= 1 --> dx2d[mask1d] returns a 2D array first. Then [y1d_mask] just used as row index not column
#    This suppose to work but not, probably because dx2d[mask1d] returns a new array not a view. So any changes/writes will not
#    take effects on the original array. But any read/print can see the original value
#    dx2d[mask1d][np.arange(y1d_mask.shape[0]), y1d_mask] -= 1
#    print(a[mask][np.arange(ymask.shape[0]), ymask])
#    Therefore only this works, even though mask1d and y1d_mask has different length, but it's fine because mask1d is boolean array
#    if it were a int array, then they must have the same length
    dx2d[mask1d, y1d_mask] -= 1 
    
    dx = dx2d.reshape(x.shape) / N
    
    return loss, dx