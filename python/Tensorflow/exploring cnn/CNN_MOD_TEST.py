# Testing single Conv NN
# 
# ==============================================================================

import input_data_window_large
import CNN_MOD
import CNN_MOD_2
import CNN_MOD_3
import CNN_STATIC_VARIABLES
import numpy as np

class CNN_TEST(object):
   """docstring for CNN_H"""
   def __init__(self, network_type, index, complete_set, window, input_size, conv_f_1, conv_f_2, nn, filter_type):
      self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
      subject_set = self.VARS.get_subject_set(False)

      
      if network_type=='original':
         output = 10
         remove_activities = self.VARS.CONVERTION_ORIGINAL_INVERSE
         keep_activities = self.VARS.CONVERTION_ORIGINAL
         self.config = self.VARS.get_config(input_size, 10, index, 100, network_type, conv_f_1, conv_f_2, nn, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, 10, remove_activities, True, keep_activities, window)

      if network_type=='sd':
         remove_activities = self.VARS.CONVERTION_STATIC_DYNAMIC_INVERSE
         keep_activities = self.VARS.CONVERTION_STATIC_DYNAMIC
         output = 2
         self.config = self.VARS.get_config(input_size, output, index, 100, network_type, conv_f_1, conv_f_2, nn, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, output, remove_activities, True, keep_activities, window)


      if network_type=='stand-up':
         remove_activities = self.VARS.CONVERTION_STAND_UP_INVERSE
         keep_activities = self.VARS.CONVERTION_STAND_UP
         self.config = self.VARS.get_config(input_size, len(keep_activities), index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, True, keep_activities, window)

      if network_type=='stairs-walk':
         remove_activities = self.VARS.CONVERTION_STAIRS_WALK_INVERSE
         keep_activities = self.VARS.CONVERTION_STAIRS_WALK
         self.config = self.VARS.get_config(input_size, len(keep_activities), index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, True, keep_activities, window)

      if network_type=='stairs':
         remove_activities = self.VARS.CONVERTION_STAIRS_INVERSE
         keep_activities = self.VARS.CONVERTION_STAIRS
         self.config = self.VARS.get_config(input_size, len(keep_activities), index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, True, keep_activities, window)

      if network_type=='cycling-sitting':
         remove_activities = self.VARS.CONVERTION_CYCLING_SITTING_INVERSE
         keep_activities = self.VARS.CONVERTION_CYCLING_SITTING
         self.config = self.VARS.get_config(input_size, len(keep_activities), index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, True, keep_activities, window)


      if network_type=='stand-nonvig-shuf':
         remove_activities = self.VARS.CONVERTION_STAND_NONVIG_SHUF_INVERSE
         keep_activities = self.VARS.CONVERTION_STAND_NONVIG_SHUF
         self.config = self.VARS.get_config(input_size, len(keep_activities), index, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, True, keep_activities, window)


      self.cnn = CNN_MOD_3.CNN_MOD(self.config)
      self.cnn.set_data_set(self.data_set)

      self.cnn.load_model('models/' + network_type+ '_' + str(input_size) + '_' + str(conv_f_1) + '_' + str(conv_f_2) + '_' + str(nn[0]) + '_' + str(nn[1]) + '_' + filter_type)
      if complete_set==1:
         print self.cnn.test_network()
      elif complete_set==2:
         print self.cnn.test_real_accuracy_on_network(self.data_set.test,window,input_size/6,convertion)
      elif complete_set == 3:
         print "Activity accuracy"
         ''' Get the original data set - 3ee what activities fails '''
         remove_activities = self.VARS.CONVERTION_ORIGINAL_INVERSE
         keep_activities = self.VARS.CONVERTION_ORIGINAL
         config = self.VARS.get_config(input_size, 10, index, 100, network_type, conv_f_1, conv_f_2, nn, filter_type)
         original_data_set = input_data_window_large.read_data_sets_without_activity(subject_set, 10, remove_activities, True, keep_activities, window)
         ''' Get the act'''
         activity_accuracy = self.cnn.get_activity_list_accuracy(original_data_set, self.data_set)
         for i in range(0, len(activity_accuracy)):
            print str(activity_accuracy[i]).replace(".",",")
      else:
         data = self.data_set.test.next_data_label(index)
         print data
         print np.argmax(data[1])+1, self.cnn.run_network(data)
      

cnn_h = CNN_TEST('sd', 2000, 1, '1.0', 600, 20, 40, [200,100], "VALID")