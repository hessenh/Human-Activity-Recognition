import input_data_window_large
import CNN
import CNN_STATIC_VARIABLES
import CNN_TRAIN


class CNN_EM(object):
		def __init__(self, network_type, iterations, window, input_size):
		self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
		subject_set = self.VARS.get_subject_set()
		self.cnn_train = CNN_TRAIN(network_type, iterations, window, input_size)
		self.cnn_test = CNN_TEST(network_type, iterations, True, window, input_size)
	
		self.config = self.VARS.get_config(input_size, 2, iterations, 100, network_type)
		convertion = self.VARS.CONVERTION_STATIC_DYNAMIC
		print 'Creating data set'
		self.data_set = read_EM_data_set(subject_set, output_size, train_remove_activities, train_convert, test_remove_activties, window):

		#Har et self.data_set._test
		#Har et self.data_set._train
		continue_EM=True

		while continue_EM==True:
			
			#TRAIN CNN MODEL WITH DATASET.TRAIN
			self.cnn_train = CNN.CNN_TWO_LAYERS(self.config)
			self.cnn_train.set_data_set(self.data_set_train)
			self.cnn_train.train_network()
			self.cnn_train.save_model('models/' + network_type +'_'+ str(input_size))

			#TEST CNN MODEL WITH DATASET.TEST
			self.cnn_test = CNN.CNN_TWO_LAYERS(config)
      		

     		self.cnn.load_model('models/' + network_type + '_' + str(input_size))
     		print self.cnn_test.test_network()
      		

         	




