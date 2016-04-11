# Hierachy of Convolutional Neural Networks
# 
# ==============================================================================
import numpy as np

class CNN_STATIC_VARS(object):
	''' Variables '''


	CONVERTION_ORIGINAL = {1:1, 2:2, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8, 11:8, 13:9, 14:10}
	CONVERTION_ORIGINAL_INVERSE = {3:3, 9:9, 12:12, 15:15, 16:16, 17:17}


	CONVERTION_STATIC_DYNAMIC = {1:1, 2:2, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8, 11:8, 13:9, 14:10}
	CONVERTION_STATIC_DYNAMIC_INVERSE = {3:3, 9:9, 12:12, 15:15, 16:16, 17:17}


	CONVERTION_STATIC = {6:1, 7:2, 8:3, 11:4}
	RE_CONVERTION_STATIC = {1:6, 2:7, 3:8, 4:11}

	CONVERTION_DYNAMIC = {1:1, 2:2, 3:3, 4:4, 5:5, 9:6, 10:7, 12:8, 13:9, 14:10, 15:11, 16:12, 17:13}
	RE_CONVERTION_DYNAIC = {1:1, 2:2, 3:3, 4:4, 5:5, 6:9, 7:10, 8:12, 9:13, 10:14, 11:15, 12:16, 13:17}
	REMOVE_DYNAMIC_ACTIVITIES = {1:1, 2:2, 3:3, 4:4, 5:5, 9:9, 10:10, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}

	CONVERTION_SHUF_STAND = {1:1,3:2,6:3}
	CONVERTION_SHUF_STAND_INVERSE = {2:2,4:4, 5:5, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}
	RE_CONVERTION_SHUF_STAND_WALK = {1:1, 2:3, 3:6}

	CONVERTION_STAND_SIT = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:2, 8:2, 10:1, 11:1, 13:2, 14:2, 16:1}
	CONVERTION_STAND_SIT_INVERSE = {9:9, 12:12, 15:15, 17:17}

	CONVERTION_LYING = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:2, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:1, 17:1}

	CONVERTION_STAND_UP = {1:1,2:2,3:3,4:4,5:5,6:6,16:7,17:8}
	CONVERTION_STAND_UP_INVERSE = {7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14}
	
	CONVERTION_STAIRS_WALK = {1:1,4:2,5:3}
	CONVERTION_STAIRS_WALK_INVERSE = {2:2, 3:3, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}

	CONVERTION_STAND_NONVIG_SHUF = {3:1,6:2,17:3}
	CONVERTION_STAND_NONVIG_SHUF_INVERSE = {1:1,2:2,4:4,5:5 ,7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16}

	CONVERTION_CYCLING_SITTING = {7:1,13:2,14:3}
	CONVERTION_CYCLING_SITTING_INVERSE = {1:1, 2:2, 3:3, 4:4, 5:5 ,6:6, 8:8, 9:9, 10:10, 11:11, 12:12, 15:15, 16:16, 17:17}

	CONVERTION_CYCLING_SITTING_LYING = {7:1,8:2,13:3,14:4}
	CONVERTION_CYCLING_SITTING_LYING_INVERSE = {1:1, 2:2, 3:3, 4:4, 5:5 ,6:6, 9:9, 10:10, 11:11, 12:12, 15:15, 16:16, 17:17}

	''' Methods '''

	''' Get length of convertion list - aka number of labels '''
	def len_convertion_list(self, convertion_list):
		return len(set(convertion_list.values()))

	''' Get config for CNN '''
	def get_config(self, input_size, output_size, iterations, batch_size, model_name, conv_list, neural_list, filter_type):
		conv_list_formated = "conv_" + format_list(conv_list)
		neural_list_formated = "conv_" + format_list(neural_list)
		return 	{
		   'input_size': input_size, # Number of inputs 
		   'output_size': output_size, # Number of ouptuts
		   'iteration_size': iterations, # Number of training iterations
		   'batch_size': batch_size, # Number of samples in each training iteration (batch)
		   'model_name': model_name + '_' + str(input_size) + '_' + conv_list_formated + neural_list_formated + filter_type + '_' + str(iterations),
		   'conv_list': conv_list,
		   'neural_list': neural_list,
		   'filter_type': filter_type
		}

	''' Subject set '''

	

	def get_subject_set(self, random):
		SUBJECTS = ["01A","02A","03A","04A"]#,"05A","06A","08A","09A","10A","11A","12A","13A","14A","15A","16A","18A","19A","20A","21A","22A","23A"]
		if random:
			SUBJECTS = ["01A","02A","03A","04A","05A","06A","08A","09A","10A","11A","12A","13A","14A","15A","16A","18A","19A","20A","21A","22A","23A"]
			TEST_SUBJECTS = []
			# Test subjects is 1/3 of subject set
			SUBJECTS_TEST_LENGTH = len(SUBJECTS) / 3
			for i in range(0,SUBJECTS_TEST_LENGTH):
				r = np.random.randint(len(SUBJECTS))
				TEST_SUBJECTS.append(SUBJECTS.pop(r))
			TRAIN_SUBJECTS = SUBJECTS
		else:
			TRAIN_SUBJECTS = ['01A', '02A', '04A', '20A', '06A', '08A', '09A', '11A', '12A', '13A', '15A', '16A', '19A', '23A']
			TEST_SUBJECTS = ['21A', '05A', '14A', '18A', '03A', '22A', '10A']
		return [TRAIN_SUBJECTS, TEST_SUBJECTS]

def format_list(list_input):
	string_list = ""
	for s in list_input:
		string_list +=str(s)+'_'
	return string_list