from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


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
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = np.tanh(prev_h.dot(Wh) + x.dot(Wx) + b) #NxH
    cache = x, Wx, Wh, b, prev_h, next_h
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
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    x, Wx, Wh, b, prev_h, next_h = cache
    
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    dhraw = dnext_h * (1 - next_h**2) #NxH
    db = np.sum(dhraw, axis=0) 
    dx = dhraw.dot(Wx.T)
    dprev_h = dhraw.dot(Wh.T)
    dWx = (x.T).dot(dhraw)
    dWh = (prev_h.T).dot(dhraw)
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
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, h_storage, cache = None, [], []
    N, T, D = x.shape
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    h_step = h0
    for step in range(T): 
        h_step, _ = rnn_step_forward(x[:, step, :], h_step, Wx, Wh, b)
        h_storage.append(h_step) # Don't need to stack h0 or initial memory
    h = np.stack(h_storage, axis=1)
    cache = (x, Wx, Wh, b, h0, h) #h0 is previous memory (N,H) , h is next memory (N,T,H)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    x, Wx, Wh, b, h0, h = cache
    N, T, H = dh.shape 
    _, _, D = x.shape
    dx, dh0, dWx, dWh, db = np.zeros((N, T, D)), np.zeros((N, H)), np.zeros((D, H)), np.zeros((H, H)), np.zeros((H))
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    
    dprev_h = np.zeros((N, H)) # This needs to be accumulated from the upstream gradient dh[:, step, :]
    next_h = h[:, T - 1, :] # Next memory state is the last item of the array.
    
    # print('h0 is ', h0)
    # print('h[:, 0, :] is', h[:, 0, :])
    
    for step in reversed(range(T)): # Backprop in a reverse order of time step.
        # Step = T-1, T-2, T-3, ..., 0. Important: h[:, 0, :] = h1
        
        # h0 is the initial memory (when time_step=0) but not included in the h array, which includes memories of ALL other timesteps
        if step == 0:
            prev_h = h0
        else:
            prev_h = h[:, step - 1, :] 
            # Extract memory of the time-step before. h_t/next_h = h[:, step, :] and h_(t-1)/prev_h = h[:, step-1, :]
            
        cache_step = x[:, step, :], Wx, Wh, b, prev_h, next_h # For each time step, the cache needs prev_h and next_h
        dnext_h = dh[:, step, :] + dprev_h # The upstream gradient is accumulated at each time step
        
        dx_step, dprev_h, dWx_step, dWh_step, db_step = rnn_step_backward(dnext_h, cache_step)
        dx[:, step, :] = dx_step
        dWx += dWx_step
        dWh += dWh_step
        db += db_step
        
        next_h = prev_h # Loop to the next memory state, in the backward direction
        
    dh0 = dprev_h # The final backward loop is the gradient of the initial memory state (h0)
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
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N, T = x.shape
    V, D = W.shape
    x_1hot = np.zeros((N, T, V))
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    cache = (x, W)
    out = W[x] #From (V, D) to (N, T, D). Each element of x becomes an row index to extract the D-vector
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
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
    x, W = cache
    dW = np.zeros_like(W)
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    
    # np.add.at allows dout to flow back only the row index, indicated by each element of the x matrix
    # The flown-back dout will match exactly the row index if dW                                        
    
    np.add.at(dW, x, dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


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
    next_h, next_c, cache = None, None, None
    N, D = x.shape
    H, _ = Wh.shape
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    a = x.dot(Wx) + prev_h.dot(Wh) + b #Nx4H
    a_i = a[:, 0:H] # Each of these is NxH. Split the big matrix into 4 smaller matrix with equal number of columns
    a_f = a[:, H:(2*H)]
    a_o = a[:, (2*H):(3*H)]
    a_g = a[:, (3*H):(4*H)]
    # print(a_i.shape, a_f.shape, a_o.shape, a_g.shape)
    i = sigmoid(a_i) # All these are also of NxH shape
    f = sigmoid(a_f)
    o = sigmoid(a_o)
    g = np.tanh(a_g) 
    next_c = f * prev_c + i * g # Element-wise multiplication. Result is NxH 
    next_h = o * np.tanh(next_c) # NxH
    cache = x, prev_h, prev_c, Wx, Wh, b, a, a_i, a_f, a_o, a_g, i, f, o, g, next_c
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
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    x, prev_h, prev_c, Wx, Wh, b, a, a_i, a_f, a_o, a_g, i, f, o, g, next_c = cache
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    do = dnext_h * np.tanh(next_c)
    dnext_c += dnext_h * o * (1 - np.tanh(next_c) ** 2)
    
    dprev_c = f * dnext_c
    df = prev_c * dnext_c
    di = g * dnext_c
    dg = i * dnext_c
    
    da_g = dg * (1 - np.tanh(a_g) ** 2)
    da_o = do * o * (1 - o)
    da_f = df * f * (1 - f)
    da_i = di * i * (1 - i)
    
    da = np.concatenate((da_i, da_f, da_o, da_g), axis=1) #Nx4H
    db = np.sum(da, axis=0) 
    dx = da.dot(Wx.T)
    dWx = (x.T).dot(da)
    dprev_h = da.dot(Wh.T)
    dWh = (prev_h.T).dot(da) 
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
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    N, T, D = x.shape
    _, H = h0.shape
    h, cache = np.zeros((N, T, H)), []
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    prev_h, prev_c = h0, np.zeros((N, H))
    for step in range(T):
        next_h, next_c, cache_step = lstm_step_forward(x[:, step, :], prev_h, prev_c, Wx, Wh, b)
        h[:, step, :] = next_h
        cache.append(cache_step)
        
        # To next loop
        prev_h = next_h
        prev_c = next_c
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    N, T, H = dh.shape
    _, D = cache[T - 1][0].shape
    dx, dh0, dWx, dWh, db = np.zeros((N, T, D)), np.zeros((N, H)), np.zeros((D, 4*H)), np.zeros((H, 4*H)), np.zeros((4*H))
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    
    # Everything in reverse order
    dnext_h = dh[:, T - 1, :] 
    dnext_c = np.zeros((N, H))
    for step in reversed(range(T)):
        cache_step = cache[step]
        dx_step, dprev_h, dprev_c, dWx_step, dWh_step, db_step = lstm_step_backward(dnext_h, dnext_c, cache_step)
        dx[:, step, :] = dx_step
        dWx += dWx_step
        dWh += dWh_step
        db += db_step 
    
        # For the next loop
        if step == 0:
            dh0 = dprev_h # No need to accumulate further because dh[:, 0, :] is dh1 not dh0
        dnext_h = dh[:, step - 1, :] + dprev_h # Accumulate the dprev_h with the previous one in dh. Has to be step-1 
        dnext_c = dprev_c
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
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
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
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
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
