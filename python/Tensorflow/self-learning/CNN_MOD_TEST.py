# Testing single Conv NN
# 
# ==============================================================================

import input_data_window_large
import CNN_MOD
import CNN_STATIC_VARIABLES
import numpy as np

class CNN_TEST(object):
   """docstring for CNN_H"""
   def __init__(self, network_type, index, complete_set, window, input_size, conv_f_1, conv_f_2, nn_1, filter_type):
      self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
      subject_set = self.VARS.get_subject_set()

      
      convertion = self.VARS.CONVERTION_STATIC_DYNAMIC
      config = self.VARS.get_config(input_size, 2, index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
      print 'Creating data set'
      self.data_set = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None, window)
      

      self.cnn = CNN_MOD.CNN_MOD(config)
      self.cnn.set_data_set(self.data_set)

      self.cnn.load_model('models/' + network_type+ '_' + str(input_size) + '_' + str(conv_f_1) + '_' + str(conv_f_2) + '_' + str(nn_1) + '_' + filter_type)
      if complete_set==1:
         print self.cnn.test_network()
      elif complete_set==2:
         print self.cnn.test_real_accuracy_on_network(self.data_set.test,window,input_size/6,convertion)
      else:
         data = self.data_set.test.next_data_label(index)
         print data
         print np.argmax(data[1])+1, self.cnn.run_network(data)
      

cnn_h = CNN_TEST('sd', 2000, 2, '1.0', 600, 20, 40, 200, "VALID")