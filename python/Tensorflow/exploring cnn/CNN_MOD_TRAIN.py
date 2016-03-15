import input_data_window_large
import CNN_STATIC_VARIABLES
import CNN_MOD
import CNN_MOD_2
import CNN_MOD_3


class CNN_MOD_TRAIN(object):
   """docstring for CNN_H"""
   def __init__(self, network_type, iterations, window, input_size, conv_f_1, conv_f_2, nn, filter_type):
      self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
      subject_set = self.VARS.get_subject_set(False)
      
      if network_type=='original':
         remove_activities = self.VARS.CONVERTION_ORIGINAL_INVERSE
         keep_activities = self.VARS.CONVERTION_ORIGINAL
         output = 10
         self.config = self.VARS.get_config(input_size, 10, iterations, 100, network_type, conv_f_1, conv_f_2, nn, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, 10, remove_activities, None, keep_activities, window)

      if network_type=='sd':
         remove_activities = self.VARS.CONVERTION_STATIC_DYNAMIC_INVERSE
         keep_activities = self.VARS.CONVERTION_STATIC_DYNAMIC
         output = 10
         self.config = self.VARS.get_config(input_size, output, iterations, 100, network_type, conv_f_1, conv_f_2, nn, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, output, remove_activities, None, keep_activities, window)

      if network_type=='stand-sit':
         remove_activities = self.VARS.CONVERTION_STAND_SIT_INVERSE
         keep_activities = self.VARS.CONVERTION_STAND_SIT
         self.config = self.VARS.get_config(input_size, 2, iterations, 100, network_type, conv_f_1, conv_f_2, nn, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, 2, remove_activities, None, keep_activities, window)

      if network_type=='stairs':
         remove_activities = self.VARS.CONVERTION_STAIRS_INVERSE
         keep_activities = self.VARS.CONVERTION_STAIRS
         self.config = self.VARS.get_config(input_size, len(keep_activities), iterations, 100, network_type, conv_f_1, conv_f_2, nn, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)

      if network_type=='stairs-walk':
         remove_activities = self.VARS.CONVERTION_STAIRS_WALK_INVERSE
         keep_activities = self.VARS.CONVERTION_STAIRS_WALK
         self.config = self.VARS.get_config(input_size, len(keep_activities), iterations, 100, network_type, conv_f_1, conv_f_2, nn, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)

      if network_type=='cycling-sitting':
         remove_activities = self.VARS.CONVERTION_CYCLING_SITTING_INVERSE
         keep_activities = self.VARS.CONVERTION_CYCLING_SITTING
         self.config = self.VARS.get_config(input_size, len(keep_activities), iterations, 100, network_type, conv_f_1, conv_f_2, nn, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)

      if network_type=='cycling-sitting-lying':
         remove_activities = self.VARS.CONVERTION_CYCLING_SITTING_LYING_INVERSE
         keep_activities = self.VARS.CONVERTION_CYCLING_SITTING_LYING
         self.config = self.VARS.get_config(input_size, len(keep_activities), iterations, 100, network_type, conv_f_1, conv_f_2, nn, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)


      if network_type=='stand-nonvig-shuf':
         remove_activities = self.VARS.CONVERTION_STAND_NONVIG_SHUF_INVERSE
         keep_activities = self.VARS.CONVERTION_STAND_NONVIG_SHUF
         self.config = self.VARS.get_config(input_size, len(keep_activities), iterations, 100, network_type, conv_f_1, conv_f_2, nn, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)


      self.cnn = CNN_MOD_4.CNN_MOD(self.config)
      self.cnn.set_data_set(self.data_set)
      
      self.cnn.train_network()
      self.cnn.save_model('models/' + network_type + '_' + str(input_size) + '_' + str(conv_f_1) + '_' + str(conv_f_2) + '_' + str(nn[0]) + '_' + str(nn[1]) + '_' + filter_type)
      self.cnn.test_network()



cnn_h = CNN_MOD_TRAIN('sd', 1000 , "1.0", 600, 20, 40, [200,100], "VALID") 


