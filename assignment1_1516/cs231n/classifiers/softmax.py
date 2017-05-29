import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
	"""
	Softmax loss function, naive implementation (with loops)

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
	# Initialize the loss and gradient to zero.
	num_train = X.shape[0]
	num_class = W.shape[1]
	loss = 0.0
	dW = np.zeros_like(W)

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using explicit loops.     #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	scores = X.dot(W)
	for i in range(num_train):
		max_score = np.max(scores[i, :])
		exp_score = np.exp(scores[i, :] - max_score)
		denom = np.sum(exp_score)
		for j in range(num_class):
			h = np.exp(scores[i][j] - max_score) / denom
			if (j == y[i]):
				loss += (-1) * np.log(h)
				dW[:, j] += (-1) * X[i, :] * (1 - h)
			else:
				dW[:, j] += (-1) * X[i, :] * (-h) 
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	loss /= num_train
	loss += (0.5) * reg * np.sum(W*W)

	dW /= num_train
	dW += reg * W

	return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
	"""
	Softmax loss function, vectorized version.

	Inputs and outputs are the same as softmax_loss_naive.
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)
	num_train = X.shape[0]

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using no explicit loops.  #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	scores = X.dot(W)
	max_scores= np.amax(scores, axis=1)
	exp_scores = np.exp(scores - np.asmatrix(max_scores).T)
	sum_vector = np.sum(exp_scores, axis=1)
	probs = exp_scores/sum_vector
	log_probs = np.log(probs)
	boolean_mat = np.zeros_like(log_probs)
	boolean_mat[np.arange(boolean_mat.shape[0]), y] = 1
	loss = np.sum(np.multiply(log_probs, boolean_mat))

	scale_matrix = (-1)*probs
	scale_matrix[np.arange(scale_matrix.shape[0]), y] = 1 + scale_matrix[np.arange(scale_matrix.shape[0]), y]
	dW = X.T.dot(scale_matrix)
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	loss /= (-1) * num_train
	loss += (0.5) * reg * np.sum(W*W)

	dW /= (-1) * num_train
	dW += reg * W

	return loss, dW

