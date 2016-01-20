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

		print 'Loading Static dynamic network'
		''' STATIC DYNAMIC GRAPH'''
		config = self.VARS.get_config(self.input_size, 2, 10, 100, 'sd')
		self.cnn_sd = CNN.CNN_TWO_LAYERS(config)
		self.cnn_sd.load_model('models/sd_' + str(self.input_size))

		''' STATIC GRAPH'''
		print 'Loading static network'
		config = self.VARS.get_config(self.input_size, 5, 10, 100, 'static')
		self.cnn_static = CNN.CNN_TWO_LAYERS(config)
		self.cnn_static.load_model('models/static_' + str(self.input_size))

		''' DYNAMIC GRAPH'''
		print 'Loading dynamic network'
		config = self.VARS.get_config(self.input_size, 12, 10, 100, 'dynamic')
		self.cnn_dynamic = CNN.CNN_TWO_LAYERS(config)
		self.cnn_dynamic.load_model('models/dynamic_' + str(self.input_size))

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
		actual_sd = np.zeros(2)
		convert =  self.VARS.CONVERTION_STATIC_DYNAMIC.get(original)
		actual_sd[convert-1] = 1

		''' STATIC DYNAMIC PREDICTION'''
		prediction_sd = self.cnn_sd.run_network_return_probability(data)
		''' DYNAMIC PREDICTION'''
		prediction_dynamic = self.cnn_dynamic.run_network_return_probability(data)
		''' STATIC PREDICTION'''
		prediction_static = self.cnn_static.run_network_return_probability(data)

		return original, prediction_sd, actual_sd, prediction_dynamic, prediction_static

	def run_network_probability(self, save=None):
		size = len(self.data_set_ORIGINAL.test.labels)
		originals = np.zeros(size)

		predictions_sd = np.zeros((size,2))
		actuals_sd = np.zeros((size, 2))

		predictions_dynamic = np.zeros((size,12))
		predictions_static = np.zeros((size,5))

		score = 0
		for i in range(0, size):
			original, prediction_sd, actual_sd, prediction_dynamic, prediction_static = self.classify_instance_probability(i)
			originals[i] = original
			actuals_sd[i] = actual_sd
			predictions_sd[i] = prediction_sd
			predictions_dynamic[i] = prediction_dynamic
			predictions_static[i] = prediction_static

		if save:
			print 'Saving predictions and results'
			np.savetxt('predictions/actual_sd_prob.csv', actuals_sd, delimiter=",")
			np.savetxt('predictions/prediction_sd_prob.csv', predictions_sd, delimiter=",")
			np.savetxt('predictions/prediction_dynamic_prob.csv', predictions_dynamic, delimiter=",")
			np.savetxt('predictions/prediction_static_prob.csv', predictions_static, delimiter=",")
			np.savetxt('predictions/original.csv', originals, delimiter=",")
		else:
			return originals,actuals_sd, predictions_sd, prediction_dynamic, predictions_static

cnn_h = CNN_H('1.5', 900)
cnn_h.initialize_networks()
print cnn_h.run_network_probability(True)