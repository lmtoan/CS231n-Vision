import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  delta = 1
  for i in xrange(num_train):
    scores = X[i].dot(W) #Score of training example i. n*(k+1) * (k+1)*C = n*C
    correct_class_score = scores[y[i]] #Equivalent to s_yi. Pin-point value of the column that has label y[i]
    count = 0
    for j in xrange(num_classes): #Within the same row of example i, loop through all the calculated scores of all C classes
      if j == y[i]: 
        continue
      margin = scores[j] - correct_class_score + delta
      if margin > 0:
        loss += margin
        count = count + 1
        dW[:, j] += X[i, :].T # Accumulate for all n examples 
    dW[:, y[i]] += (-1) * count * X[i, :].T 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  delta = 1.0
  dW = np.zeros(W.shape) # Initialize the gradient as zero
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) # NxC
  #print scores[0, :]
  #print y[0]
  correct_scores = scores[np.arange(num_train), y]
  #print correct_scores.shape
  #print correct_scores[0]
  margins = np.maximum(0, scores - np.asmatrix(correct_scores).T + delta) # Compare with matrix of zeros
  margins[np.arange(num_train), y] = 0 # Don't accumulate the column of deltas in the correct class
  loss = np.sum(margins)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  tracking_matrix = (margins > 0).astype(int)
  count = np.sum(tracking_matrix, axis=1) # Across columns. Sum of all incorrect classes across each example
  tracking_matrix[np.arange(tracking_matrix.shape[0]), y.flatten()] = (-1) * count.flatten() # For every example/row, change the item in correct-label column into -count of that row
  dW = (X.T).dot(tracking_matrix) #Each column of X.T is an example i, will be multiplied with tracker-column to accumulate gradient
  
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
