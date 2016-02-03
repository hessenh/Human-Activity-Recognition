# Hierachy of Convolutional Neural Networks
# 
# ==============================================================================


class CNN_STATIC_VARS(object):
	''' Variables '''
	CONVERTION_STATIC_DYNAMIC = {1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:1, 10:1, 11:2, 12:1, 13:1, 14:1, 15:1, 16:1, 17:1}

	CONVERTION_ORIGINAL = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}

	CONVERTION_STATIC = {6:1, 7:2, 8:3, 11:4}
	RE_CONVERTION_STATIC = {1:6, 2:7, 3:8, 4:11}

	CONVERTION_DYNAMIC = {1:1, 2:2, 3:3, 4:4, 5:5, 9:6, 10:7, 12:8, 13:9, 14:10, 15:11, 16:12, 17:13}
	RE_CONVERTION_DYNAIC = {1:1, 2:2, 3:3, 4:4, 5:5, 6:9, 7:10, 8:12, 9:13, 10:14, 11:15, 12:16, 13:17}
	REMOVE_DYNAMIC_ACTIVITIES = {1:1, 2:2, 3:3, 4:4, 5:5, 9:9, 10:10, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}

	CONVERTION_SHUF_STAND = {1:1,3:2,6:3}
	CONVERTION_SHUF_STAND_INVERSE = {2:2,4:4, 5:5, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}
	RE_CONVERTION_SHUF_STAND_WALK = {1:1, 2:3, 3:6}

	CONVERTION_STAND_SIT = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:2, 8:2, 9:3, 10:3, 11:3, 12:1, 13:3, 14:3, 15:3, 16:3, 17:3}

	CONVERTION_LYING = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:2, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:1, 17:1}

	''' Methods '''

	''' Get length of convertion list - aka number of labels '''
	def len_convertion_list(self, convertion_list):
		return len(set(convertion_list.values()))

	''' Get config for CNN '''
	def get_config(self, input_size, output_size, iterations, batch_size, model_name, conv_f_1, conv_f_2, nn_1, filter_type):
		return 	{
		   'input_size': input_size, # Number of inputs 
		   'output_size': output_size, # Number of ouptuts
		   'iteration_size': iterations, # Number of training iterations
		   'batch_size': batch_size, # Number of samples in each training iteration (batch)
		   'model_name': model_name + '_' + str(input_size) + '_' + str(conv_f_1) + '_' + str(conv_f_2) + '_' + str(nn_1) + '_' + filter_type,
		   'conv_f_1': conv_f_1,
		   'conv_f_2': conv_f_2,
		   'nn_1': nn_1,
		   'filter_type': filter_type
		}

	''' Subject set '''
	def get_subject_set(self):
		TRAIN_SUBJECTS = ["01A"]#,"02A","07A","08A","09A","10A","11A","12A","13A","14A","15A","16A","18A","19A","21A","22A","23A"]
		TEST_SUBJECTS = ["03A"]#,"04A","05A"]#,"04A","05A"]#,"P04","P06","P07","P08","P09","P10","P14","P15","P16","P17","P18","P19","P20","P21"]
		return [TRAIN_SUBJECTS, TEST_SUBJECTS]
