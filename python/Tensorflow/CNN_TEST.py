# Testing single Conv NN
# 
# ==============================================================================

import input_data_window_large
import CNN
import CNN_STATIC_VARIABLES
import numpy as np

class CNN_TEST(object):
   """docstring for CNN_H"""
   def __init__(self, network_type, index, complete_set, window, input_size):
      self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
      subject_set = self.VARS.get_subject_set()

      if network_type == 'sd':
         convertion = self.VARS.CONVERTION_STATIC_DYNAMIC
         config = self.VARS.get_config(input_size, 2, index, 100, network_type)
         print 'Creating data set'
         self.data_set = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None, window)
      
      if network_type == 'original':
         convertion = self.VARS.CONVERTION_ORIGINAL
         config = self.VARS.get_config(input_size, 17, index, 100, network_type)
         print 'Creating data set'
         self.data_set = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None, window)
      
      if network_type == 'static':
         remove_activities = self.VARS.REMOVE_DYNAMIC_ACTIVITIES
         keep_activities = self.VARS.CONVERTION_STATIC
         config = self.VARS.get_config(input_size, 5, index, 100, network_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, self.VARS.len_convertion_list(keep_activities), remove_activities, None, keep_activities, window)

      if network_type == 'dynamic':
         remove_activities = self.VARS.CONVERTION_STATIC
         keep_activities = self.VARS.CONVERTION_DYNAMIC
         config = self.VARS.get_config(input_size, 12, index, 100, network_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, self.VARS.len_convertion_list(keep_activities), remove_activities, None, keep_activities, window)

      if network_type == 'shuf_stand':
         remove_activities = self.VARS.CONVERTION_SHUF_STAND_INVERSE
         keep_activities = self.VARS.CONVERTION_SHUF_STAND
         config = self.VARS.get_config(input_size, 3, index, 100, network_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)


      self.cnn = CNN.CNN_TWO_LAYERS(config)
      self.cnn.set_data_set(self.data_set)

      self.cnn.load_model('models/' + network_type + '_' + str(input_size))
      if complete_set:
         print self.cnn.test_network()
      else:
         data = self.data_set.test.next_data_label(index)
         print np.argmax(data[1])+1, self.cnn.run_network(data)
      

cnn_h = CNN_TEST('sd', 2000, True, '1.5', 900)