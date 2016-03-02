import input_data_window_large
import CNN_STATIC_VARIABLES
import CNN_MOD
import CNN_MOD_2


class CNN_MOD_TRAIN(object):
   """docstring for CNN_H"""
   def __init__(self, network_type, iterations, window, input_size, conv_f_1, conv_f_2, nn, filter_type):
      self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
      subject_set = self.VARS.get_subject_set(False)
      
      if network_type=='original':
         convertion = self.VARS.CONVERTION_ORIGINAL
         self.config = self.VARS.get_config(input_size, self.VARS.len_convertion_list(convertion), iterations, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         print 'Creating data set'
         self.data_set = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None, window)
      
      if network_type=='stand-up':
         remove_activities = self.VARS.CONVERTION_STAND_UP_INVERSE
         keep_activities = self.VARS.CONVERTION_STAND_UP
         self.config = self.VARS.get_config(input_size, len(keep_activities), iterations, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)

      if network_type=='stairs':
         remove_activities = self.VARS.CONVERTION_STAIRS_INVERSE
         keep_activities = self.VARS.CONVERTION_STAIRS
         self.config = self.VARS.get_config(input_size, len(keep_activities), iterations, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)

      if network_type=='stairs-walk':
         remove_activities = self.VARS.CONVERTION_STAIRS_WALK_INVERSE
         keep_activities = self.VARS.CONVERTION_STAIRS_WALK
         self.config = self.VARS.get_config(input_size, len(keep_activities), iterations, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)

      if network_type=='cycling-sitting':
         remove_activities = self.VARS.CONVERTION_CYCLING_SITTING_INVERSE
         keep_activities = self.VARS.CONVERTION_CYCLING_SITTING
         self.config = self.VARS.get_config(input_size, len(keep_activities), iterations, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)


      if network_type=='stand-nonvig-shuf':
         remove_activities = self.VARS.CONVERTION_STAND_NONVIG_SHUF_INVERSE
         keep_activities = self.VARS.CONVERTION_STAND_NONVIG_SHUF
         self.config = self.VARS.get_config(input_size, len(keep_activities), iterations, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)


      self.cnn = CNN_MOD.CNN_MOD(self.config)
      self.cnn.set_data_set(self.data_set)
      self.cnn.train_network()
      self.cnn.save_model('models/' + network_type + '_' + str(input_size) + '_' + str(conv_f_1) + '_' + str(conv_f_2) + '_' + str(nn[0]) + '_' + str(nn[1]) + '_' + filter_type)



cnn_h = CNN_MOD_TRAIN('cycling-sitting', 3000 , "1.0", 600, 20, 40, 200, "SAME") 

