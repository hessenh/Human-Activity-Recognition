# Segmenting network
# 
# ==============================================================================

import input_data_window_large
import CNN_MOD
import CNN_STATIC_VARIABLES
import numpy as np



class CNN_SEGMENT(object):
	"""docstring for CNN_SEGMENT"""
	def __init__(self, network_type, window, input_size, conv_f_1, conv_f_2, nn_1, filter_type):
		
		''' CNN INIT'''
		self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
		subject_set = self.VARS.get_subject_set()

		if network_type == 'sd':
			convertion = self.VARS.CONVERTION_STATIC_DYNAMIC
			config = self.VARS.get_config(input_size, 2, 1, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
			self.data_set = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None, window)

		self.cnn = CNN_MOD.CNN_MOD(config)
		self.cnn.set_data_set(self.data_set)
		self.cnn.load_model('models/' + network_type+ '_' + str(input_size) + '_' + str(conv_f_1) + '_' + str(conv_f_2) + '_' + str(nn_1) + '_' + filter_type)

		''' Segmenting part '''
		# Get the predictions and save them
		window_list = []
		data_length =  self.data_set.test._num_examples
		for i in range(0,data_length):
			data = self.data_set.test.next_data_label(i)
			prediction = self.cnn.run_network([data[0]])
			window_list.append(prediction)
		
		window_size = input_size / 6
		# After an activity, add a timestamp
		segment_list = [0]
		for i in range(0, len(window_list)-1):
			if window_list[i] != window_list[i+1]:
				segment_list.append((i+1)*int(window_size/2))
		segment_list.append(len(window_list)*int(window_size/2))
		print segment_list

			

CNN_SEGMENT('sd','0.76', 456, 10, 20, 1024, "SAME")