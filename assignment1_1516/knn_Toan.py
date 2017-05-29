import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

show_image = 0
if(show_image != 0):
	# Visualize 15 examples from each of the 10 classes
	classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	num_classes = len(classes)
	samples_per_class = 6
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

# Subsample the data for more efficient code execution in this exercise 
num_training = 5000
mask = range(num_training) #range(x): list out all numbers from 1 to x 
X_train = X_train[mask]
y_train = y_train[mask]
num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]
print 'Test Print: ', y_train[range(10)] #Print out a vector of 1st 10 elements of y_train

# Reshape to 2D
X_train = np.reshape(X_train, (X_train.shape[0], -1)) 
X_test = np.reshape(X_test, (X_test.shape[0], -1)) 
print X_train.shape, X_test.shape

from cs231n.classifiers import KNearestNeighbor
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

dists = classifier.compute_distances_two_loops(X_test) 
print 'Distance Matrix: ', dists.shape
dists_show = 0
if(dists_show != 0):
	plt.imshow(dists, interpolation='none')
	plt.show()

# Now implement the function predict_labels and run the code below: # We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=3)
# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test * 100
print 'Got %d / %d correct, and accuracy: %f' % (num_correct, num_test, accuracy)

# Now lets speed up distance matrix computation by using partial vectorization
dists_one = classifier.compute_distances_one_loop(X_test)
# To ensure that our vectorized implementation is correct, we make sure that # agrees with the naive implementation. There are many ways to decide whethe # two matrices are similar; one of the simplest is the Frobenius norm. In ca # you haven't seen it before, the Frobenius norm of two matrices is the squa # root of the squared sum of differences of all elements; in other words, re # the matrices into vectors and compute the Euclidean distance between them. difference = np.linalg.norm(dists - dists_one, ord='fro')
difference = np.linalg.norm(dists - dists_one, ord='fro')
print 'Difference was: %f' % (difference, ) 
if difference < 0.001:
	print 'Good! The distance matrices are the same' 
else:
	print 'Uh-oh! The distance matrices are different'

# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'

# Let's compare how fast the implementations are
def time_function(f, *args):
  """
  Call a function f with args and return the time (in seconds) that it took to execute.
  """
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print 'Two loop version took %f seconds' % two_loop_time

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print 'One loop version took %f seconds' % one_loop_time

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print 'No loop version took %f seconds' % no_loop_time

# you should see significantly faster performance with the fully vectorized implementation



