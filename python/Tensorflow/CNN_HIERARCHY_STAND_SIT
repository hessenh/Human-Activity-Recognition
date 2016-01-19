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
		convertion = self.VARS.CONVERTION_ORIGINAL
		self.data_set_ORIGINAL = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None, window)
		
	def initialize_networks(self):
		''' ORIGINAL GRAPH'''
		print 'Loading original network'
		config = self.VARS.get_config(self.input_size, 17, 10, 100, 'original')
		self.cnn_original = CNN.CNN_TWO_LAYERS(config)
		self.cnn_original.set_data_set(self.data_set_ORIGINAL)
		#self.cnn_original.load_model('models/original')

		print 'Loading stand sit network'
		''' STATIC DYNAMIC GRAPH'''
		config = self.VARS.get_config(self.input_size, 3, 10, 100, 'stand_sit')
		self.cnn_stand_sit = CNN.CNN_TWO_LAYERS(config)
		self.cnn_stand_sit.load_model('models/stand_sit_' + str(self.input_size))

		

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
		''' Convert actual to sd format'''
		actual_stand_sit = np.zeros(3)
		convert =  self.VARS.CONVERTION_STAND_SIT.get(original)
		actual_stand_sit[convert-1] = 1

		''' STATIC DYNAMIC PREDICTION'''
		prediction_stand_sit = self.cnn_stand_sit.run_network_return_probability(data)
		

		return original, prediction_stand_sit, actual_stand_sit

	def run_network_probability(self, save=None):
		size = len(self.data_set_ORIGINAL.test.labels)
		originals = np.zeros(size)

		predictions_stand_sit = np.zeros((size,3))
		actuals_stand_sit = np.zeros((size, 3))

		score = 0
		for i in range(0, size):
			original, prediction_stand_sit, actual_stand_sit = self.classify_instance_probability(i)
			originals[i] = original
			actuals_stand_sit[i] = actual_stand_sit
			predictions_stand_sit[i] = prediction_stand_sit

		if save:
			print 'Saving predictions and results'
			np.savetxt('predictions/actual_stand_sit_prob.csv', actuals_stand_sit, delimiter=",")
			np.savetxt('predictions/prediction_stand_sit_prob.csv', predictions_stand_sit, delimiter=",")
			np.savetxt('predictions/original.csv', originals, delimiter=",")
		else:
			return originals,actuals_stand_sit, predictions_stand_sit

cnn_h = CNN_H('1.5', 900)
cnn_h.initialize_networks()
print cnn_h.run_network_probability(True)