# Hierachy of Convolutional Neural Networks
# 
# ==============================================================================


class CNN_STATIC_VARS(object):
	''' Variables '''
	CONVERTION_STATIC_DYNAMIC = {1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:1, 10:2, 11:2, 12:1, 13:1, 14:1, 15:1, 16:1, 17:1}

	CONVERTION_ORIGINAL = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}

	CONVERTION_STATIC = {6:1, 7:2, 8:3, 10:4, 11:5}
	RE_CONVERTION_STATIC = {1:6, 2:7, 3:8, 4:10, 5:11}

	CONVERTION_DYNAMIC = {1:1, 2:2, 3:3, 4:4, 5:5, 9:6, 12:7, 13:8, 14:9, 15:10, 16:11,17:12}
	RE_CONVERTION_DYNAIC = {1:1, 2:2, 3:3, 4:4, 5:5, 6:9, 7:12, 8:13, 9:14, 10:15, 11:16 , 12:17}
	REMOVE_DYNAMIC_ACTIVITIES = {1:1, 2:2, 3:3, 4:4, 5:5, 9:9, 12:12, 13:13, 14:14, 15:15, 16:16 , 17:17}

	CONVERTION_SHUF_STAND = {1:1,3:2,6:3}
	CONVERTION_SHUF_STAND_INVERSE = {2:2,4:4, 5:5, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}
	RE_CONVERTION_SHUF_STAND_WALK = {1:1, 2:3, 3:6}

	CONVERTION_STAND_SIT = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:2, 8:2, 9:3, 10:3, 11:3, 12:3, 13:2, 14:1, 15:1, 16:3, 17:3}


	''' Methods '''

	''' Get length of convertion list - aka number of labels '''
	def len_convertion_list(self, convertion_list):
		return len(set(convertion_list.values()))

	''' Get config for CNN '''
	def get_config(self, input_size, output_size, iterations, batch_size, model_name):
		return 	{
		   'input_size': input_size, # Number of inputs 
		   'output_size': output_size, # Number of ouptuts
		   'iteration_size': iterations, # Number of training iterations
		   'batch_size': batch_size, # Number of samples in each training iteration (batch)
		   'model_name': model_name
		}

	''' Subject set '''
	def get_subject_set(self):
		TRAIN_SUBJECTS = ["01A"] #,"03A","04A","05A","07A","08A","09A","10A","11A","12A","13A","14A","15A","16A","18A","19A","21A","22A","23A"]
		TEST_SUBJECTS = ["02A"]
		return [TRAIN_SUBJECTS, TEST_SUBJECTS]
