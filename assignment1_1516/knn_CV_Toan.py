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

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# Split the training set into 5 folds. These 5 folds now become Cross-Validation set. 4 folds will be for train_cv. 5th fold will be for test_cv
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in k_choices:
	k_to_accuracies[k] = [] #Establish each item of the k_to_accuracies array is an empty list
	print 'Evaluating at k = %d' %k 

	for j in range(num_folds): #Loop through all the folds of the training data. CV-fold is j-th. Other folds for training
		X_test_cv = X_train_folds[j]
		y_test_cv = y_train_folds[j]
		#print 'Test CV: ', X_test_cv.shape, y_test_cv.shape

		X_train_cv = np.vstack(X_train_folds[0:j] + X_train_folds[j+1:]) #Leaving out the j-th array. X/y_train_folds are LISTs
		y_train_cv = np.hstack(y_train_folds[0:j] + y_train_folds[j+1:])
		#print 'Train CV: ', X_train_cv.shape, y_train_cv.shape

		classifier.train(X_train_cv, y_train_cv)
		dists_cv = classifier.compute_distances_no_loops(X_test_cv)
		#print 'Dists CV: ', dists_cv.shape
		y_test_pred = classifier.predict_labels(dists_cv, k)
		num_correct_cv = np.sum(y_test_pred == y_test_cv)
		accuracy_cv = float(num_correct_cv)/y_test_cv.shape[0]
		print y_test_cv.shape[0]
		print 'Accuracy at %d-nearest neighbors, cv-fold is %d-th fold, is %.2f' %(k, j+1, accuracy_cv*100)
		
		k_to_accuracies[k].append(accuracy_cv)

################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
best_k = 1
max_accuracy = 0
for k in sorted(k_to_accuracies):
	for accuracy in k_to_accuracies[k]:
		if(accuracy > max_accuracy):
			max_accuracy = accuracy
			best_k = k

# plot the raw observations
for k in k_choices:
	accuracies = k_to_accuracies[k]
	plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
print 'Performing best %d-nearest neighbors classifier...' %best_k
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
