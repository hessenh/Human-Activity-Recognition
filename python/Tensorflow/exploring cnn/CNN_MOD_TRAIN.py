import input_data_window_large
import CNN_STATIC_VARIABLES
import CNN_MOD


class CNN_MOD_TRAIN(object):
   """docstring for CNN_H"""
   def __init__(self, network_type, iterations, window):
      self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
      subject_set = self.VARS.get_subject_set()
      
      self.config = self.VARS.get_config(576, 17, iterations, 100, network_type)
      convertion = self.VARS.CONVERTION_ORIGINAL
      print 'Creating data set'
      self.data_set = input_data_window_large.read_data_sets(subject_set, 17, convertion, None, window)


      self.cnn = CNN_MOD.CNN_MOD(self.config)
      self.cnn.set_data_set(self.data_set)
      self.cnn.train_network()
      self.cnn.save_model('models/' + network_type)

cnn_h = CNN_MOD_TRAIN('original', 1000 , '0.96') #'0.96'
#0.665
#0.742