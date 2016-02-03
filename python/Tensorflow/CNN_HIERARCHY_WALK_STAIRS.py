# Hierachy of Convolutional Neural Networks
# 
# ==============================================================================

import input_data_window_large
import CNN
import CNN_STATIC_VARIABLES
import copy
import numpy as np

class CNN_H(object):
	"""docstring for CNN_H"""
	def __init__(self, window, input_size):
		self.input_size = input_size
		self.window = window
		self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
		subject_set = self.VARS.get_subject_set()

		print 'Creating ORIGINAL data set'
		keep_activities = self.VARS.CONVERTION_ORIGINAL
		remove_activities = { 2:2, 3:3, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}
		self.data_set_ORIGINAL = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)
		
	def initialize_networks(self):
		''' ORIGINAL GRAPH'''
		print 'Loading original network'
		config = self.VARS.get_config(self.input_size, 17, 10, 100, 'original')
		self.cnn_original = CNN.CNN_TWO_LAYERS(config)
		self.cnn_original.set_data_set(self.data_set_ORIGINAL)
		#self.cnn_original.load_model('models/original')

		''' WALK STAIRS GRAPH'''
		print 'Loading static network'
		config = self.VARS.get_config(self.input_size, 3, 10, 100, 'walk_stairs')
		print "done with get config"
		self.cnn_walk_stairs = CNN.CNN_TWO_LAYERS(config)
		print "initialize walk stairs networks"
		self.cnn_walk_stairs.load_model('models/walk_stairs_' + str(self.input_size))
		print "Load walk stair model"
		

	def classify_instance(self, index):
		''' Test classifiers on one data index, returns the actual and prediction label'''

		''' ORIGINAL '''
		data = self.data_set_ORIGINAL.test.next_data_label(index)
		actual = np.argmax(data[1])+1
		#print 'Actual', actual

		''' STATIC DYNAMIC PREDICTION'''
		prediction = self.cnn_sd.run_network(data)
		#print "SD prediction", prediction
		if prediction == 2:
			''' STATIC PREDICTION '''
			prediction = self.cnn_static.run_network(data)
			prediction = self.VARS.RE_CONVERTION_STATIC.get(prediction)
			#print "STATIC prediction", prediction
		else:
			''' DYNAMIC PREDICTION '''
			prediction = self.cnn_dynamic.run_network(data)
			prediction = self.VARS.RE_CONVERTION_DYNAIC.get(prediction)
			#print "DYNAMIC prediction",prediction

		return actual, prediction
			#return actual == prediction

	def run_network(self, save=None):
		''' Test whole subject and return actual and predictions lists'''
		size = len(self.data_set_ORIGINAL.test.labels)
		predictions = np.zeros(size)
		actuals = np.zeros(size)

		score = 0
		for i in range(0, size):
			actual, prediction = self.classify_instance(i)
			actuals[i] = actual
			predictions[i] = prediction
			if actual == prediction:
				score += 1
		print 'Accuracy', score*1.0 / size
		if save:
			print 'Saving predictions and results'
			np.savetxt('predictions/actual.csv', actuals, delimiter=",")
			np.savetxt('predictions/prediction.csv', predictions, delimiter=",")
		else:
			return actuals, predictions


	''' Probability classifiers '''
	def classify_instance_probability(self, index):
		''' Test classifiers on one data index, returns the actual and probability prediction '''

		''' ORIGINAL '''
		data = self.data_set_ORIGINAL.test.next_data_label(index)
		actual = data[1]
		original =  np.argmax(actual) + 1
		if original == 5:


		''' Convert actual to sd format'''
		actual_walk_stairs = np.zeros(3)
		convert =  self.VARS.CONVERTION_WALK_STAIRS.get(original)
		actual_walk_stairs[convert-1] = 1

		''' DYNAMIC PREDICTION'''
		prediction_walk_stairs = self.cnn_walk_stairs.run_network_return_probability(data)
		
		

		return original, actual_walk_stairs, prediction_walk_stairs

	def run_network_probability(self, save=None):
		size = len(self.data_set_ORIGINAL.test.labels)
		originals = np.zeros(size)

		predictions_walk_stairs = np.zeros((size,3))
		actuals_walk_stairs = np.zeros((size, 3))

		score = 0
		for i in range(0, size):
			original, actual_walk_stairs, prediction_walk_stairs= self.classify_instance_probability(i)
			originals[i] = original
			actuals_walk_stairs[i] = actual_walk_stairs
			predictions_walk_stairs[i] = prediction_walk_stairs

		if save:
			print 'Saving predictions and results'
			np.savetxt('predictions/actual_walk_stairs_prob.csv', actuals_walk_stairs, delimiter=",")
			np.savetxt('predictions/prediction_walk_stairs_prob.csv', predictions_walk_stairs, delimiter=",")
			np.savetxt('predictions/original.csv', originals, delimiter=",")
		else:
			return originals, predictions_walk_stairs

cnn_h = CNN_H('0.96', 576)
cnn_h.initialize_networks()
print cnn_h.run_network_probability(True)