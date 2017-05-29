import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet

def rel_error(x, y):
	""" returns relative error """
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
	np.random.seed(0)
	return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
	np.random.seed(1)
	X = 10 * np.random.randn(num_inputs, input_size)
	y = np.array([0, 1, 2, 2, 1])
	return X, y

net = init_toy_model()
X, y = init_toy_data()

print 'X:', X
print 'Y:', y

scores = net.loss(X)
print 'Your scores:'
print scores
print '----**----'
print 'Correct scores:'
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print correct_scores
print '----**----'

# The difference should be very small. We get < 1e-7
print 'Difference between your scores and correct scores:'
print np.sum(np.abs(scores - correct_scores))

loss, _ = net.loss(X, y, reg=0.1)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print 'Difference between your loss and correct loss:'
print np.sum(np.abs(loss - correct_loss))

from cs231n.gradient_check import eval_numerical_gradient

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, y, reg=0.1)

# these should all be less than 1e-8 or so
for param_name in grads:
	f = lambda W: net.loss(X, y, reg=0.1)[0]
	param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
	print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))

net = init_toy_model()
stats = net.train(X, y, X, y, learning_rate=1e-1, reg=1e-5, num_iters=500, verbose=False)

print 'Final training loss: ', stats['loss_history'][-1]

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
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

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.5, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print 'Validation accuracy: ', val_acc

show_graph = 0
if (show_graph != 0):
	# Plot the loss function and train / validation accuracies
	plt.subplot(2, 1, 1)
	plt.plot(stats['loss_history'])
	plt.title('Loss history')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	plt.subplot(2, 1, 2)
	plt.plot(stats['train_acc_history'], label='train')
	plt.plot(stats['val_acc_history'], label='val')
	plt.title('Classification accuracy history')
	plt.xlabel('Epoch')
	plt.ylabel('Clasification accuracy')
	plt.show()

from cs231n.vis_utils import visualize_grid

# Visualize the weights of the network

def show_net_weights(net):
  W1 = net.params['W1']
  W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
  plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
  plt.gca().axis('off')
  plt.show()

# show_net_weights(net)
results = {}
best_val = -1
best_net = None # store the best model into this 

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
print '-------------_*****--------------'
print 'Tuning Parameters: '
for lr in np.arange(0.001, 0.05, 0.002):
	for reg in np.arange(0.1, 1, 0.1):
		for hid in np.arange(50, 155, 20):
			print 'At hidden size: %d, LR: %f, and REG: %f' % (hid, lr, reg)
			net_iter = TwoLayerNet(input_size, hid, num_classes)
			stats_iter = net_iter.train(X_train, y_train, X_val, y_val, num_iters=2500, batch_size=200, learning_rate=lr, learning_rate_decay=0.95, reg=reg, verbose=True)
			y_val_pred_iter = net_iter.predict(X_val)
			y_train_pred_iter = net_iter.predict(X_train)
			val_acc = np.mean(y_val == y_val_pred_iter)
			train_acc = np.mean(y_train == y_train_pred_iter)
			results[(lr, reg)] = (train_acc, val_acc) # Turple Mapping
			print 'Validation accuracy: %f' % (val_acc)
			if (val_acc > best_val):
				best_val = val_acc
				best_net = net_iter
				print 'Best so far: %f' % (val_acc)
###################################################

#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################

# Print out results.
for lr, reg in sorted(results):
	train_accuracy, val_accuracy = results[(lr, reg)]
	print 'With lr %e reg %e train accuracy: %f val accuracy: %f' % (
				lr, reg, train_accuracy, val_accuracy)
	
print 'Best validation accuracy achieved during cross-validation: %f' % best_val

# Visualize the cross-validation results
import math
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results] # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()

# Evaluate the best svm on test set
y_test_pred = best_net.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print '2-layer NN on raw pixels final test set accuracy: %f' % test_accuracy
