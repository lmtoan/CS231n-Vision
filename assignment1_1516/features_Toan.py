# Goal: we can improve our classification performance by training linear classifiers not on raw pixels 
# but on features that are computed from the raw pixels.

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

from cs231n.features import color_histogram_hsv, hog_feature

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
  # Load the raw CIFAR-10 data
  cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  
  # Subsample the data
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]

  return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

# Extract Features
# For each image we will compute a Histogram of Oriented Gradients (HOG) as well as a color histogram using the hue channel in HSV color space. 
# We form our final feature vector for each image by concatenating the HOG and color histogram feature vectors.
# Roughly speaking, HOG should capture the texture of the image while ignoring color information, and the color histogram represents the color of the input image while ignoring texture. 
# As a result, we expect that using both together ought to work better than using either alone. Verifying this assumption would be a good thing to try for the bonus section.
# The hog_feature and color_histogram_hsv functions both operate on a single image and return a feature vector for that image. 
# The extract_features function takes a set of images and a list of feature functions and evaluates each feature function on each image, 
# storing the results in a matrix where each column is the concatenation of all feature vectors for a single image.

from cs231n.features import *

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

print 'Sizes: ', X_train_feats.shape, y_train.shape, X_val_feats.shape, y_val.shape, X_test_feats.shape, y_test.shape

# Use the validation set to tune the learning rate and regularization strength

from cs231n.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [1e5, 1e6, 1e7]

results = {}
best_val = -1
best_svm = None

tune_SVM = 0
if (tune_SVM != 0):
	################################################################################
	# TODO:                                                                        #
	# Use the validation set to set the learning rate and regularization strength. #
	# This should be identical to the validation that you did for the SVM; save    #
	# the best trained classifer in best_svm. You might also want to play          #
	# with different numbers of bins in the color histogram. If you are careful    #
	# you should be able to get accuracy of near 0.44 on the validation set.       #
	################################################################################
	print '-------------_*****--------------'
	print 'Tuning Parameters for SVM: '
	for lr in np.arange(0.0001, 0.001, 0.0001): #(0.0000001, 0.000001, 0.0000001)
		for reg in np.arange(0.1, 1, 0.1): # (1e4, 1e5, 20000)
			print 'LR: %f and REG: %d' % (lr, reg)
			svm_iter = LinearSVM()
			svm_iter.train(X_train_feats, y_train, learning_rate=lr, reg=reg, num_iters=3000, verbose=True)
			y_val_pred_iter = svm_iter.predict(X_val_feats)
			y_train_pred_iter = svm_iter.predict(X_train_feats)
			val_acc = np.mean(y_val == y_val_pred_iter)
			train_acc = np.mean(y_train == y_train_pred_iter)
			results[(lr, reg)] = (train_acc, val_acc) # Turple Mapping
			print 'Validation accuracy: %f' % (val_acc)
			if (val_acc > best_val):
				best_val = val_acc
				best_svm = svm_iter
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

	# Evaluate your trained SVM on the test set
	y_test_pred = best_svm.predict(X_test_feats)
	test_accuracy = np.mean(y_test == y_test_pred)
	print test_accuracy

	# An important way to gain intuition about how an algorithm works is to
	# visualize the mistakes that it makes. In this visualization, we show examples
	# of images that are misclassified by our current system. The first column
	# shows images that our system labeled as "plane" but whose true label is
	# something other than "plane".

	examples_per_class = 8
	classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	for cls, cls_name in enumerate(classes):
	    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
	    idxs = np.random.choice(idxs, examples_per_class, replace=False)
	    for i, idx in enumerate(idxs):
	        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
	        plt.imshow(X_test[idx].astype('uint8'))
	        plt.axis('off')
	        if i == 0:
	            plt.title(cls_name)
	plt.show()

print X_train_feats.shape

from cs231n.classifiers.neural_net import TwoLayerNet

input_dim = X_train_feats.shape[1]
hidden_dim = [300,400,500,600,700]
num_classes = 10

best_net = None

################################################################################
# TODO: Train a two-layer neural network on image features. You may want to    #
# cross-validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################
print '-------------_*****--------------'
print 'Tuning Parameters for Neural Nets: '
for lr in np.arange(0.5, 2, 0.1):
	for reg in np.arange(0.001, 0.01, 0.002):
		for hid in hidden_dim:
			print 'At hidden size: %d, LR: %f, and REG: %f' % (hid, lr, reg)
			net_iter = TwoLayerNet(input_dim, hid, num_classes)
			stats_iter = net_iter.train(X_train_feats, y_train, X_val_feats, y_val, num_iters=3000, batch_size=200, learning_rate=lr, learning_rate_decay=0.95, reg=reg, verbose=True)
			y_val_pred_iter = net_iter.predict(X_val_feats)
			y_train_pred_iter = net_iter.predict(X_train_feats)
			val_acc = np.mean(y_val == y_val_pred_iter)
			train_acc = np.mean(y_train == y_train_pred_iter)
			print 'Validation accuracy: %f' % (val_acc)
			if (val_acc > best_val):
				best_val = val_acc
				best_net = net_iter
				print 'Best so far: %f' % (val_acc)
################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# Run your neural net classifier on the test set. You should be able to
# get more than 55% accuracy.

test_acc = (net.predict(X_test_feats) == y_test).mean()
print test_acc

