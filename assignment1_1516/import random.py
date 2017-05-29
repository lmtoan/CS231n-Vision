# In this exercise you will:
# - implement a fully-vectorized loss function for the SVM
# - implement the fully-vectorized expression for its analytic gradient
# - check your implementation using numerical gradient
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
print ('Raw Train data shape: ', X_train.shape)
print ('Raw Train labels shape: ', y_train.shape)
print ('Raw Test data shape: ', X_test.shape)
print ('Raw Test labels shape: ', y_test.shape)

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
print ('--------------****----------------')
print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print ('Validation data shape: ', X_val.shape)
print ('Validation labels shape: ', y_val.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)
print ('Dev data shape: ', X_dev.shape)
print ('Dev labels shape: ', y_dev.shape)


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

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

# Evaluate the naive implementation of the loss we provided for you:
from cs231n.classifiers.linear_svm import svm_loss_naive
import time

# generate a random SVM weight matrix of small numbers
W = np.random.randn(3073, 10) * 0.0001 

loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)
print('loss: %f' % (loss, ))
