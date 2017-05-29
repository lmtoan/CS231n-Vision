# Softmax exercise
# You will:
# - implement a fully-vectorized loss function for the Softmax classifier
# - implement the fully-vectorized expression for its analytic gradient
# - check your implementation with numerical gradient
# - use a validation set to tune the learning rate and regularization strength
# - optimize the loss function with SGD
# - visualize the final learned weights

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
print 'Raw Train data shape: ', X_train.shape
print 'Raw Train labels shape: ', y_train.shape
print 'Raw Test data shape: ', X_test.shape
print 'Raw Test labels shape: ', y_test.shape

show_image = 0
if(show_image != 0):
	# Visualize some examples from the dataset.
	# We show a few examples of training images from each class.
	classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	num_classes = len(classes)
	samples_per_class = 7
	for y, cls in enumerate(classes):
		idxs = np.flatnonzero(y_train == y)
		idxs = np.random.choice(idxs, samples_per_class, replace=False)
		for i, idx in enumerate(idxs):
			plt_idx = i * num_classes + y + 1
			plt.subplot(samples_per_class, num_classes, plt_idx)
			plt.imshow(X_train[idx].astype('uint8'))
			plt.axis('off')
			if i == 0:
				plt.title(cls)
	plt.show()

# Split the data into train, val, and test sets. In addition we will create a small development set as 
# a subset of the training data; we can use this for development so our code runs faster.
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Our validation set will be num_validation points from the original training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make a development set, which is a small subset of the training set.
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# We use the first num_test points of the original test set as our test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
print '--------------****----------------'
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape
print 'Dev data shape: ', X_dev.shape
print 'Dev labels shape: ', y_dev.shape


# Preprocessing
# Compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
if(show_image != 0):
	print(mean_image[:10]) # print a few of the elements
	plt.figure(figsize=(4,4))
	plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
	plt.show()

# Subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

# Append the bias dimension of ones (i.e. bias trick) so that our SVM only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print 'After adding bias dimension: ', X_train.shape, X_val.shape, X_test.shape, X_dev.shape

from cs231n.classifiers.softmax import softmax_loss_naive
import time

# Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(3073, 10) * 0.0001
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

# As a rough sanity check, our loss should be something close to -log(0.1).
print 'loss: %f' % loss
print 'sanity check: %f' % (-np.log(0.1))

# Complete the implementation of softmax_loss_naive and implement a (naive)
# version of the gradient that uses nested loops.
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

# As we did for the SVM, use numeric gradient checking as a debugging tool.
# The numeric gradient should be close to the analytic gradient.
from cs231n.gradient_check import grad_check_sparse
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

# similar to SVM case, do another gradient check with regularization
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 1e2)
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 1e2)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

# Now that we have a naive implementation of the softmax loss function and its gradient,
# implement a vectorized version in softmax_loss_vectorized.
# The two versions should compute the same results, but the vectorized version should be
# much faster.
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.00001)
toc = time.time()
print 'naive loss: %e computed in %fs' % (loss_naive, toc - tic)

from cs231n.classifiers.softmax import softmax_loss_vectorized
tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.00001)
toc = time.time()
print 'vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic)

# As we did for the SVM, we use the Frobenius norm to compare the two versions
# of the gradient.
grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print 'Loss difference: %f' % np.abs(loss_naive - loss_vectorized)
print 'Gradient difference: %f' % grad_difference

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.
from cs231n.classifiers import Softmax
results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7, 5e-7]
regularization_strengths = [5e4, 1e8]

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained softmax classifer in best_softmax.                          #
################################################################################
print '-------------_*****--------------'
print 'Tuning Parameters: '
for lr in np.arange(0.0000001, 0.0000005, 0.00000005):
	for reg in np.arange(5e4, 1e8, 20000):
		sm_iter = Softmax()
		sm_iter.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=3000, verbose=True)
		y_val_pred_iter = sm_iter.predict(X_val)
		y_train_pred_iter = sm_iter.predict(X_train)
		val_acc = np.mean(y_val == y_val_pred_iter)
		train_acc = np.mean(y_train == y_train_pred_iter)
		results[(lr, reg)] = (train_acc, val_acc) # Turple Mapping
		print 'Validation accuracy: %f' % (val_acc)
		if (val_acc > best_val):
			best_val = val_acc
			best_softmax = sm_iter
			print 'Best so far: %f' % (val_acc)
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
	
# Print out results.
for lr, reg in sorted(results):
	train_accuracy, val_accuracy = results[(lr, reg)]
	print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
				lr, reg, train_accuracy, val_accuracy)
	
print 'best validation accuracy achieved during cross-validation: %f' % best_val

# evaluate on test set
# Evaluate the best softmax on test set
y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print 'softmax on raw pixels final test set accuracy: %f' % (test_accuracy, )

# Visualize the learned weights for each class
w = best_softmax.W[:-1,:] # strip out the bias
w = w.reshape(32, 32, 3, 10)

w_min, w_max = np.min(w), np.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in xrange(10):
	plt.subplot(2, 5, i + 1)
	# Rescale the weights to be between 0 and 255
	wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
	plt.imshow(wimg.astype('uint8'))
	plt.axis('off')
	plt.title(classes[i])

	