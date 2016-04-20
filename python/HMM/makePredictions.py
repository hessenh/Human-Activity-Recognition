import input_data_window_large
import CNN_MOD_2
import CNN_MOD_4
import CNN_STATIC_VARIABLES
import numpy as np

class CNN_H(object):
   	"""docstring for CNN_H"""
	def __init__(self, network_type, index, window, input_size,  conv_layers, neural_layers, filter_type):
		self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
		subject_set = self.VARS.get_subject_set(False)

		if network_type == 'original':
			remove_activities = self.VARS.CONVERTION_ORIGINAL_INVERSE
			keep_activities = self.VARS.CONVERTION_ORIGINAL
			self.config = self.VARS.get_config(input_size, 10, index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
			self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, 10, remove_activities, None, keep_activities, window)

		if network_type == 'sd':
			remove_activities = self.VARS.CONVERTION_STATIC_DYNAMIC_INVERSE
			keep_activities = self.VARS.CONVERTION_STATIC_DYNAMIC
			self.config = self.VARS.get_config(input_size, 10, index, 100, network_type,  conv_layers, neural_layers, filter_type)
			print 'Creating data set'
			self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, 13, remove_activities, None, keep_activities, window)
		

		if network_type == 'stand-sit':
			remove_activities = self.VARS.CONVERTION_STAND_SIT_INVERSE
			keep_activities = self.VARS.CONVERTION_STAND_SIT
			self.config = self.VARS.get_config(input_size, 2, index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
			self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, 2, remove_activities, None, keep_activities, window)

		if network_type =='stairs':
			remove_activities = self.VARS.CONVERTION_STAIRS_INVERSE
			keep_activities = self.VARS.CONVERTION_STAIRS
			self.config = self.VARS.get_config(input_size, len(keep_activities), index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
			self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)

		if network_type =='stairs-walk':
			remove_activities = self.VARS.CONVERTION_STAIRS_WALK_INVERSE
			keep_activities = self.VARS.CONVERTION_STAIRS_WALK
			self.config = self.VARS.get_config(input_size, len(keep_activities), index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
			self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)

		if network_type=='stand-nonvig-shuf':
			remove_activities = self.VARS.CONVERTION_STAND_NONVIG_SHUF_INVERSE
			keep_activities = self.VARS.CONVERTION_STAND_NONVIG_SHUF
			self.config = self.VARS.get_config(input_size, len(keep_activities), index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
			self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)

		if network_type=='cycling-sitting':
			remove_activities = self.VARS.CONVERTION_CYCLING_SITTING_INVERSE
			keep_activities = self.VARS.CONVERTION_CYCLING_SITTING
			self.config = self.VARS.get_config(input_size, 3, index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
			self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, 3, remove_activities, None, keep_activities, window)


		if network_type == 'stand-up':
			remove_activities = self.VARS.CONVERTION_STAND_UP_INVERSE
			keep_activities = self.VARS.CONVERTION_STAND_UP
			config = self.VARS.get_config(input_size, len(keep_activities), index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
			print 'Creating data set'
			self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, self.VARS.len_convertion_list(keep_activities), remove_activities, None, keep_activities, window) 



		self.cnn = CNN_MOD_4.CNN_FILTER(self.config)
		self.cnn.set_data_set(self.data_set)
		self.cnn.load_model('models/' + self.config['model_name'] )
		


	''' Probability classifiers '''
	def classify_instance_probability(self, index):
		''' Test classifiers on one data index, returns the actual and probability prediction '''

		
		data = self.data_set.test.next_data_label(index)
		actual = data[1]

		prediction = self.cnn.run_network_return_probability(data)
		

		return  prediction, actual

	def run_network_probability(self,network_type,numOfAct):
		size = len(self.data_set.test.labels)
		predictions = np.zeros((size,numOfAct))
		actuals = np.zeros((size, 13))

		score = 0
		for i in range(0, size):	
			prediction, actual = self.classify_instance_probability(i)
			actuals[i] = actual
			predictions[i] = prediction
		
		print 'Saving predictions and results'
		np.savetxt('predictions/actual_'+network_type+'_prob_test_all.csv', actuals, delimiter=",")
		np.savetxt('predictions/prediction_'+network_type+'_prob_test_all.csv', predictions, delimiter=",")
		

cnn_h = CNN_H('sd', 20000, '1.0', 600, [20, 40], [1500], "VALID")

print cnn_h.run_network_probability('sd',10)









