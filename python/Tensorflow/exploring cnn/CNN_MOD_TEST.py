# Testing single Conv NN
# 
# ==============================================================================

import input_data_window_large
import CNN_MOD
import CNN_STATIC_VARIABLES
import numpy as np

class CNN_TEST(object):
   """docstring for CNN_H"""
   def __init__(self, network_type, index, complete_set, window):
      self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
      subject_set = self.VARS.get_subject_set()

      
      convertion = self.VARS.CONVERTION_ORIGINAL
      config = self.VARS.get_config(576, 17, index, 100, network_type)
      print 'Creating data set'
      self.data_set = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None, window)
      

      self.cnn = CNN_MOD.CNN_MOD(config)
      self.cnn.set_data_set(self.data_set)

      self.cnn.load_model('models/' + network_type)
      if complete_set:
         print self.cnn.test_network()
      else:
         data = self.data_set.test.next_data_label(index)
         print np.argmax(data[1])+1, self.cnn.run_network(data)
      

cnn_h = CNN_TEST('original', 2000, True, '0.96')