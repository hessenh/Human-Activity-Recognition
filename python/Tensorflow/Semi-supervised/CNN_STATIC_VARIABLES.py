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

	''' Methods '''

	''' Get length of convertion list - aka number of labels '''
	def len_convertion_list(self, convertion_list):
		return len(set(convertion_list.values()))

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


	def get_subject_set_SS(self, random = None):
		SUBJECTS = ["01A","02A","03A","04A"]#,"05A","06A","08A","09A","10A","11A","12A","13A","14A","15A","16A","18A","19A","20A","21A","22A","23A"]
		if random:
			SUBJECTS = ["01A","02A","03A","04A"]#,"05A","06A","08A","09A","10A"]#,"11A","12A","13A","14A","15A","16A","18A","19A","20A","21A","22A","23A"]
			i = np.random.randint(len(SUBJECTS))
			TEST_SUBJECTS = [SUBJECTS.pop(i)]
			TRAIN_SUBJECTS = SUBJECTS
		else:
			TRAIN_SUBJECTS = ["01A","02A","04A","05A"]#,"06A","08A","09A"]#,"10A","11A","12A","13A","14A","15A","16A","18A","19A","20A","21A","22A","23A"]
			TEST_SUBJECTS = ["03A"]

		return [TRAIN_SUBJECTS, TEST_SUBJECTS]


def format_list(list_input):
	string_list = ""
	for s in list_input:
		string_list +=str(s)+'_'
	return string_list