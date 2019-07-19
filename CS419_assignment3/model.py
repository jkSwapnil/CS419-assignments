import tensorflow as tf
import numpy as np
import math
import random
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
# include any other imports that you want

'''
This file contains a class for you to implement your neural net.
Basic function skeleton is given, and some comments to guide you are also there.

You will find it convenient to look at the tensorflow API to understand what functions to use.
'''

'''
Implement the respective functions in this class
You might also make separate classes for separate tasks , or for separate kinds of networks (normal feed-forward / CNNs)
'''
    
class myNeuralNet:
	# you can add/modify arguments of *ALL* functions
	# you might also add new functions, but *NOT* remove these ones
	def __init__(self, dim_input_data, dim_output_data): # you can add/modify arguments of this function 
		# Using such 'self-ization', you can access these members in later functions of the class
		# You can do such 'self-ization' on tensors also, there is no change
		self.dim_input_data = dim_input_data
		self.dim_output_data = dim_output_data
		self.predictions=[]
		self.x= tf.placeholder(tf.float32, shape=(None, dim_input_data))
		self.y= tf.placeholder(tf.float32, shape=(None, dim_output_data))
		self.layers= []
		self.previous=None

		# Create placeholders for input : data as well as labels
		# You might want to initialising some container to store all the layers of the network

	def addHiddenLayer(self, layer_dim, layer_number, activation_fn=None, regularizer_fn=None ):
		if layer_number == 1:
			self.previous = self.x
		else:
			self.previous = self.layers[layer_number-2]
		self.layers.append(tf.layers.dense(self.previous, layer_dim, activation_fn))
		self.previous=self.layers[layer_number-1]
		#h1: tf.Variable(tf.random_normal([num_input, n_hidden_1]))
		# Add a layer to the network of layer_dim
		# It might be a good idea to append the new layer to the container of layers that you initialized before

	def addFinalLayer(self, activation_fn=None, regularizer_fn=None):
		# self.out_layer = tf.layers.dense(self.previous, 10)
		self.out_layer = tf.layers.dense(self.previous, self.dim_output_data)
		# We don't take layer_dim here, since the dimensionality of final layer is
		# already stored in self.dim_output_data

		# Create the output of the final layer as logits
		# You might also like to apply the final activation function (softmax / sigmoid) to get the predicted labels
	
	def setup_training(self, learn_rate):
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out_layer, labels=self.y))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
		self.train_step = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
		# Define loss, you might want to store it as self.loss
		# Define the train step as self.train_step = ..., use an optimizer from tf.train and call minimize(self.loss)

	def setup_metrics(self):
		self.predictions= tf.argmax(self.out_layer,1)
		#self.acc = tf.metrics.accuracy(labels=self.y, predictions=np.transpose(self.predictions))
		# Use the predicted labels and compare them with the input labels(placeholder defined in __init__)
		# to calculate accuracy, and store it as self.accuracy


	'''
	#------------------------Training Fnction for MNIST and Census-----------------------------------------
	def train(self, sess, max_epochs, batch_size, train_size, xtrain, ytrain, xval, yval, xtest, print_step = 100): # valid_size, test_size, etc
		# Write your training part here
		# sess is a tensorflow session, used to run the computation graph
		# note that all the functions uptil now were just constructing the computation graph
		
		# one 'epoch' represents that the network has seen the entire dataset once - it is just standard terminology
		steps_per_epoch = int(train_size/batch_size)
		#max_steps = max_epochs * steps_per_epoch
		itr=1
		for i in range(max_epochs):
			for step in range(steps_per_epoch-1):
				(batch_x, batch_y) = xtrain[step*batch_size:(step+1)*batch_size,:], ytrain[step*batch_size:(step+1)*batch_size,:]
				print("epoches, step, iteration", i, step, itr)
				# now run the train_step, self.loss on this batch of training data. something like :
				_, train_loss = sess.run([self.train_step, self.loss], feed_dict={self.x: batch_x , self.y: batch_y})
				print("training loss", train_loss)
				itr+=1
				if (itr % print_step) == 0:
					# read the validation dataset and report loss, accuracy on it by running
					val_loss = sess.run([self.loss], feed_dict={self.x: xval, self.y: yval})
					# remember that the above will give you val_acc, val_loss as numpy values and not tensors
					print("validation loss", val_loss)
			# 	store these train_loss and validation_loss in lists/arrays, write code to plot them vs steps
			# 	Above curves are *REALLY* important, they give deep insights on what's going on
			# # -- for loop ends --
			# # Now once training is done, run predictions on the test set
		test_predictions = sess.run(self.predictions, feed_dict={self.x: xtest})
		#print(len(test_predictions))
		#print(len(yval))
		return test_predictions
	'''


	# -------------------------- Training Function for Speech Recongnition---------------------------------
	def train(self, sess, max_epochs, batch_size, train_size, train_signal, train_lbls, valid_signal, valid_lbls, test_signal):
		steps_per_epoch = math.ceil(train_size/batch_size)
		max_steps = max_epochs*steps_per_epoch
		print(max_steps)
		for step in range(int(max_steps)):
			# select batch_size elements randomly from training data
			sampled_indices = random.sample(range(train_size), batch_size)
			trn_signal = train_signal[sampled_indices]
			trn_labels = train_lbls[sampled_indices]
			print("step",step)
			# now run the train_step, self.loss on this batch of training data. something like :
			_, train_loss = sess.run([self.train_step, self.loss], feed_dict={self.x: trn_signal , self.y: trn_labels})
			print("training loss", train_loss)
			if (step % steps_per_epoch) == 0:
				val_loss = sess.run([self.loss], feed_dict={self.x: valid_signal, self.y: valid_lbls})
				print(val_loss)
			#sess.run(self.train_step, feed_dict={input_data: trn_signal, input_labels: trn_labels})
		test_prediction = sess.run([self.predictions], feed_dict={self.x: test_signal})
		return test_prediction


		# This is because we will ask you to submit test_predictions, and some marks will be based on how your net performs on these unseen instances (test set)
'''

		We have done everything in train(), but
		you might want to create another function named eval(),
		which calculates the predictions on test instances ...
'''

'''
	NOTE:
	you might find it convenient to make 3 different train functions corresponding to the three different tasks,
	and call the relevant one from each train_*.py
	The reason for this is that the arguments to the train() are different across the tasks
'''
'''
	Example, for the speech part, the train() would look something like :
	(NOTE: this is only a rough structure, we don't claim that this is exactly what you have to do.)
	
	train(self, sess, batch_size, train_size, max_epochs, train_signal, train_lbls, valid_signal, valid_lbls, test_signal):
		steps_per_epoch = math.ceil(train_size/batch_size)
		max_steps = max_epochs*steps_per_epoch
		print(max_steps)
		for step in range(max_steps):
			# select batch_size elements randomly from training data
			sampled_indices = random.sample(range(train_size), batch_size)
			trn_signal = train_signal[sampled_indices]
			trn_labels = train_lbls[sampled_indices]
			if (step % steps_per_epoch) == 0:
				val_loss, val_acc = sess.run([self.loss, self.accuracy], feed_dict={input_data: valid_signal, input_labels: valid_lbls})
				print(step, val_acc, val_loss)
			sess.run(self.train_step, feed_dict={input_data: trn_signal, input_labels: trn_labels})
		test_prediction = sess.run([self.predictions], feed_dict={input_data: test_signal})
		return test_prediction
'''