from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
import csv

from model import myNeuralNet

# x denotes features, y denotes labels
xtrain = np.load('data/mnist/xtrain.npy')
ytrain = np.load('data/mnist/ytrain.npy')

xval = np.load('data/mnist/xval.npy')
yval = np.load('data/mnist/yval.npy')

xtest = np.load('data/mnist/xtest.npy')

dim_input = 784
dim_output = 10

max_epochs = 10
learn_rate = 1e-4
batch_size = 50

train_size = len(xtrain)
valid_size = len(xval)
test_size = len(xtest)
#print(train_size)

total_images = []
total_labels = []

# Create Computation Graph
nn_instance = myNeuralNet(dim_input, dim_output)
nn_instance.addHiddenLayer(1000, 1, tf.nn.relu)
nn_instance.addHiddenLayer(1000, 2, tf.nn.relu)
nn_instance.addHiddenLayer(1000, 3, tf.nn.relu)
# add more hidden layers here by calling addHiddenLayer as much as you want
# a net of depth 3 should be sufficient for most tasks
nn_instance.addFinalLayer()
nn_instance.setup_training(learn_rate)
nn_instance.setup_metrics()

# Training steps
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	test_pred = nn_instance.train(sess, max_epochs, batch_size, train_size, xtrain, ytrain, xval, yval, xtest) # fill in other arguments as you modify the train(self, sess, ...) in model.py
	test_pred= np.array(test_pred)
	np.savetxt("predict.csv", test_pred, fmt='%d')
	print(test_pred)
	print("Output saved in the file")