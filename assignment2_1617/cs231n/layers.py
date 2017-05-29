from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_inp = x.reshape(x.shape[0], -1)
    out  = x_inp.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    inp_shapes = x.shape
    
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x_inp = x.reshape(x.shape[0], -1)
    dw = (x_inp.T).dot(dout)
    db = np.sum(dout, axis=0)
    dx = ((dout).dot(w.T)).reshape(x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    mapping = (cache > 0).astype(int)
    dx = dout * mapping
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    sample_mean, sample_var, xmu, ivar = 0, 0, 0, 0
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        xmu = x - sample_mean
        ivar = 1/(np.sqrt(sample_var + eps))
        x_norm = xmu * ivar
        out = gamma * x_norm + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_norm = (x - running_mean)/(np.sqrt(running_var) + eps)
        out = gamma * x_norm + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    cache = (x, sample_mean, sample_var, xmu, ivar, x_norm, gamma, beta)
    
    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, mu, var, xmu, ivar, x_norm, gamma, beta = cache
    N, D = dout.shape 
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    # Because the summation of beta during the forward pass is a row-wise summation. 
    # During the backward pass we need to sum up the gradient over all of its columns
    dbeta = np.sum(dout, axis=0)
    
    # Element-wise multiplication of local grad (x_norm) and above grad (dout) gives NxD. 
    # We need to sum up the gradients over dimension N
    dgamma = np.sum(dout*x_norm, axis=0)
    
    """
    Refer to this computational graph: 
    https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    """
    # Backward pass through the entire normalization process
    dx_norm = dout*gamma #NxD
    # Mult gate (x-mu) * ivar. Xmu is NxD so has to collapse to (D,)
    dxmu1 = dx_norm*ivar #NxD
    divar = np.sum(dx_norm*xmu, axis=0) #D 
    # ivar gate: (var+eps)^(-0.5)
    dvar_sqrt = (-1) * (ivar**2) * divar
    dvar = (0.5) * (ivar) * dvar_sqrt #D
    # var gate: var = sum(xmu**2)/N. A column-wise summation during the forward pass, during the backward pass means that we
    # evenly distribute the gradient over all rows for each column. We want dxmu_sq has the same dimension as xmu_sq (NxD)
    dxmu_sq = (1/N) * dvar * np.ones_like(dout)
    dxmu2 = 2 * xmu * dxmu_sq #NxD
    dxmu = dxmu1 + dxmu2 
    # xmu gate. xmu = x-mu
    dx1 = dxmu
    dmu = (-1) * np.sum(dxmu, axis=0) #D
    # mu gate. mu = 1/N sum(x)
    dx2 = (1/N) * dmu * np.ones_like(dout)
    dx = dx1 + dx2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, mu, var, xmu, ivar, x_norm, gamma, beta = cache
    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_norm*dout, axis=0)
    dx = (1. / N) * gamma * ivar * (N * dout - np.sum(dout, axis=0) - xmu * ivar**2 * np.sum(dout*xmu, axis=0))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout. 
        # We drop and scale at train time and don't do anything at test time.
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p 
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param, verbose=False):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW). So dot product of (N, C, H_pad*W_pad) and (
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    stride, pad = conv_param['stride'], conv_param['pad']
    
    N, C, H, W = x.shape
    x_pad = np.lib.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant', constant_values=0)
    if verbose:
        print('X_pad', x_pad.shape)
    _, _, H_pad, W_pad = x_pad.shape
    F, _, HH, WW = w.shape
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    W_out = (W - WW + 2 * pad) / stride + 1
    H_out = (H - HH + 2 * pad) / stride + 1
    
    weight_row = np.reshape(w, (w.shape[0], -1))
    loop_count = 1
    conv_storage = []
    extract_storage = []
    pos_storage = []
    
    for start_H in range(0, H_pad, stride):
        for start_W in range(0, W_pad, stride):
            if start_H + HH > H_pad or start_W + WW > W_pad:
                break
            if verbose:
                print('Convolve Loop: ', loop_count) # Total number of loops needs to match W_out*H_out
            extract = x_pad[:, :, start_H:(start_H + HH), start_W:(start_W + WW)]
            pos = [slice(start_H, start_H + HH), slice(start_W, start_W + WW)]
            pos_storage.append(pos)
            extract = np.reshape(extract, (extract.shape[0], -1)) #N*filter_pixel. Flatten all the weight of the filter 
            extract_storage.append(extract)
            conv = extract.dot(weight_row.T) + b #N*F
            conv_storage.append(conv) 
            loop_count += 1
    
    out_conv = np.stack(conv_storage, axis=2)
    if verbose:
        print('Before-reshape Convolve', out_conv.shape)
    out = np.reshape(out_conv, (out_conv.shape[0], out_conv.shape[1], H_out, W_out))
                           
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param, extract_storage, pos_storage)
    return out, cache


def conv_backward_naive(dout, cache, verbose=False):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param, x_scanned, pos = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    dx, dw, db = np.zeros((x.shape)), np.zeros((w.shape)), np.zeros(F)
    dx_pad = np.zeros((N, C, W + 2*pad, H + 2*pad))
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    if verbose:
        print('dout: ', dout.shape)
        print('dx_pad: ', dx_pad.shape)
    dout_conv = np.reshape(dout, (dout.shape[0], dout.shape[1], -1))
    weight_row = np.reshape(w, (w.shape[0], -1))
    
    for layer in range(dout_conv.shape[2]):
        dscore = dout_conv[:, :, layer]
        dweight = (x_scanned[layer].T).dot(dscore)
        dx_scanned = dscore.dot(weight_row) 
        if verbose and layer == 0:
            print('dscore: ',dscore.shape)
            print('dweight: ',dweight.shape)
            print('dx_scanned: ',dx_scanned.shape)
            print(pos[layer][0], pos[layer][1])
            print(dx_pad[:, :, pos[layer][0], pos[layer][1]].shape)
        dw += np.reshape(dweight.T, (-1, C, HH, WW))
        db += np.sum(dscore, axis=0)
        # Flow the gradient back to dx_pad that has the same dimension as the padded-X. 
        # Accumulate the gradient of the tiny image (C,WW,HH) right at the position that the tiny image 
        # was extracted. 
        dx_pad[:, :, pos[layer][0], pos[layer][1]] += np.reshape(dx_scanned, (-1, C, HH, WW)) 
        
    # To get rid of the padding
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    
    A pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 
    along both width and height, discarding 75% of the activations
    
    Every MAX operation would in this case be taking a max over 4 numbers (little 2x2 region in some depth slice). 
    The depth dimension remains unchanged.
    """
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W = x.shape
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    H_out = int((H - pool_height) / stride + 1)
    W_out = int((W - pool_width) / stride + 1)
    out = np.zeros((N, C, H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            scan_W = slice(j*stride, j*stride + pool_width)
            scan_H = slice(i*stride, i*stride + pool_height)
            scanned = x[:, :, scan_H, scan_W]
            max_element = np.max(scanned, axis=(2,3)) #[N, C, max_element] 
            out[:, :, i, j] = max_element #[N, C, insert_spot]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    dx = np.zeros_like(x)
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    for i in range(H_out):
        for j in range(W_out):
            scan_W = slice(j*stride, j*stride + pool_width)
            scan_H = slice(i*stride, i*stride + pool_height)
            x_scanned = x[:, :, scan_H, scan_W]
            max_x_scanned = np.max(x_scanned, axis=(2,3))
            mask = (x_scanned == (max_x_scanned)[:, :, None, None]) # Pinpoint the position of the maxium element extracted 
            
            # Convert each dout element [:, :, i, j] into a true-false map that matches the dimension of the scanned-image map
            dx[:, :, scan_H, scan_W] = (dout[:, :, i, j])[:, :, None, None] * mask  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    
    Spatial batch normalization computes a mean and variance for each of the C feature channels by computing statistics 
    over both the minibatch dimension N and the spatial dimensions H and W
    
    From Reddit-user jcjohnss:
    This is a hint for spatial batch normalization: you will need to reshape numpy arrays. 
    When you do so you need to be careful and think about the order that numpy iterates over elements when reshaping.
    
    Suppose that x has shape (A, B, C) and you want to "collapse" the first and third dimensions 
    into a single dimension, resulting in an array of shape (A*C, B).
    Calling y = x.reshape(A * B, C) will give an array of the right shape, but it will be wrong. 
    This will put x[0, 0, 0] into y[0, 0], then x[0, 0, 1] into y[0, 1], etc until eventually x[0, 0, C - 1] will go to 
    y[0, C - 1] (assuming C < B); then x[0, 1, 0] will go to y[0, C]. This probably isn't the behavior you wanted.
    
    Due this order for moving elements in a reshape, the rule of thumb is that it is only safe to collapse adjacent dimension;
    reshaping (A, B, C) to (A*C, B) is unsafe since the collapsed dimensions are not adjacent. 
    To get the correct result, you should first use the transpose method to permute the dimensions so that 
    the dimensions you want to collapse are adjacent, and then use reshape to actually collapse them.
    Therefore for the above example you should call y = x.transpose(0, 2, 1).reshape(A * C, B)
    """
    out, cache = None, None
    out_storage, cache_storage = [], []
    
    N, C, W, H = x.shape
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    for img in range(N):
        image = x[img, :, :, :] # CxWxH
        spatial = np.reshape(image, (image.shape[0], -1)) #Cx(WH)
        out_spatial, cache_spatial = batchnorm_forward(spatial.T, gamma, beta, bn_param) #Cx(WH)
        out_spatial = out_spatial.T # Flip to (WH)xC to match up with beta (C,) and gamma (C,)
        out_storage.append(out_spatial.reshape((out_spatial.shape[0], W, H))) #Stack N images 
        cache_storage.append(cache_spatial)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    out = np.stack(out_storage, axis=0)
    cache = cache_storage
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape
    dx, dgamma, dbeta = None, np.zeros(C), np.zeros(C)
    dx_storage = []
    
    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    for img in range(N):
        cache_spatial = cache[img]
        dout_spatial = dout[img].reshape((dout[img].shape[0], -1)) #Cx(HW)
        dx_temp, dgamma_temp, dbeta_temp = batchnorm_backward_alt(dout_spatial.T, cache_spatial) #Incoming Grad is (HW)xC
        dgamma += dgamma_temp
        dbeta += dbeta_temp
        dx_temp = dx_temp.T
        dx_storage.append(dx_temp.reshape((dx_temp.shape[0], H, W)))
        
    dx = np.stack(dx_storage, axis=0) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
