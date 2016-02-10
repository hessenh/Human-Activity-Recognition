import input_data_window_large
import CNN_STATIC_VARIABLES
import CNN_MOD
import CNN_MOD_2


class CNN_MOD_TRAIN(object):
   """docstring for CNN_H"""
   def __init__(self, network_type, iterations, window, input_size, conv_f_1, conv_f_2, nn_1, filter_type):
      self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
      subject_set = self.VARS.get_subject_set()
      
      if network_type == 'sd':
         self.config = self.VARS.get_config(input_size, 2, iterations, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         convertion = self.VARS.CONVERTION_STATIC_DYNAMIC
         print 'Creating data set'
         self.data_set = input_data_window_large.read_data_sets(subject_set, 2, convertion, None, window)

      if network_type == 'original':
         self.config = self.VARS.get_config(input_size, 17, iterations, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         convertion = self.VARS.CONVERTION_ORIGINAL
         print 'Creating data set'
         self.data_set = input_data_window_large.read_data_sets(subject_set, 17, convertion, None, window)

      self.cnn = CNN_MOD_2.CNN_MOD(self.config)
      self.cnn.set_data_set(self.data_set)
      self.cnn.train_network()
      self.cnn.save_model('models/' + network_type + '_' + str(input_size) + '_' + str(conv_f_1) + '_' + str(conv_f_2) + '_' + str(nn_1) + '_' + filter_type)

cnn_h = CNN_MOD_TRAIN('original', 3000 , '1.0', 600, 20, 40, 200, "SAME") 

