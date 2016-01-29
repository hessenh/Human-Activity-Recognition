import input_data_window_large
import CNN
import CNN_STATIC_VARIABLES
import numpy as np


class CNN_EM(object):
	def __init__(self, network_type, iterations, window, input_size):
		self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
		subject_set = self.VARS.get_subject_set()
		transition_remove_activties = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}
		train_remove_activities = {9:9}
		if network_type == 'sd':	
			self.config = self.VARS.get_config(input_size, 2, iterations, 100, network_type)
			train_convert = self.VARS.CONVERTION_STATIC_DYNAMIC
			print 'Creating data set'
			self.data_set = input_data_window_large.read_EM_data_set(subject_set, 2, train_remove_activities, train_convert, transition_remove_activties, window)
		if network_type == 'original':
			train_convert = self.VARS.CONVERTION_ORIGINAL
			self.config = self.VARS.get_config(input_size, 17, iterations, 100, network_type)
			print 'Creating data set'
			self.data_set = input_data_window_large.read_EM_data_set(subject_set, 17, train_remove_activities, train_convert, transition_remove_activties, window)


	
		#TRAIN CNN MODEL WITH DATASET.TRAIN
		self.cnn = CNN.CNN_TWO_LAYERS(self.config)
		self.cnn.set_data_set(self.data_set)
		self.cnn.train_network()
		self.cnn.save_model('models/' + network_type +'_'+ str(input_size) + '_Without_T')

		continue_EM=1
		threshold = 0.9
		while continue_EM < 4:
			print "hei"
			above_threshold = []
			for i in range(0,len(self.data_set.transition._data)):
				''' Get the transitions data point'''
				data_batch = self.data_set.transition._data[i]
				prediction = self.cnn.run_network_return_probability([[data_batch]])[0]
				activity = np.argmax(prediction)
				if prediction[activity] >= threshold:
					above_threshold.append([i,activity])

			''' Set the newly shuffled data set '''
			self.data_set = input_data_window_large.shuffle_data(above_threshold, self.data_set)
			self.cnn.set_data_set(self.data_set)
			self.cnn.train_network()

			continue_EM+=1
			if len(above_threshold) == 0:
				continue_EM = 4
		
		self.cnn.save_model('models/' + network_type +'_'+ str(input_size) +'_EM')

cnn_h = CNN_EM('original', 1000, '1.5', 900)

