# Train different CNN
# 
# ==============================================================================

import input_data_window_large
import CNN
import CNN_STATIC_VARIABLES
		

class CNN_TRAIN(object):
	"""docstring for ClassName"""
	def __init__(self, network_type, iterations, window, input_size):
		self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
		subject_set = self.VARS.get_subject_set()
	
	
		self.config = self.VARS.get_config(input_size, 2, iterations, 100, network_type)
		convertion = self.VARS.CONVERTION_STATIC_DYNAMIC
		print 'Creating data set'
		self.data_set = input_data_window_large.read_data_sets_without_activity(subjects_set, 2, {9:9}, None, {1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 10:2, 11:2, 12:1, 13:1, 14:1, 15:1, 16:1, 17:1}, window)

		

		(subject_set, self.VARS.len_convertion_list(convertion), convertion, None, window)
		print self.data_set.train._labels
		  


		#self.cnn = CNN.CNN_TWO_LAYERS(self.config)
		#self.cnn.set_data_set(self.data_set)
		#self.cnn.train_network()
		#self.cnn.save_model('models/' + network_type +'_'+ str(input_size))


cnn_h = CNN_TRAIN('sd', 500, '1.5', 900)