import data_generator
import CNN_STATIC_VARIABLES
import CNN_MOD_2
import numpy as np

class CNN_SS_TRAIN(object):
   """docstring for CNN_H"""
   def __init__(self, network_type, iterations, window, input_size, conv_f_1, conv_f_2, nn_1, filter_type, number_of_classifiers):
      self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
      subject_set = self.VARS.get_subject_set_SS(True)
      
      if network_type == "sd":
         self.output = 2
         self.config = self.VARS.get_config(input_size, self.output, iterations, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)
         convertion = self.VARS.CONVERTION_STATIC_DYNAMIC
      if network_type == "original":
         self.output = 17
         convertion = self.VARS.CONVERTION_ORIGINAL
         self.config = self.VARS.get_config(input_size, self.output, iterations, 100, network_type, conv_f_1, conv_f_2, nn_1, filter_type)

      print 'Creating data set'
      self.data_set = data_generator.read_SS_data_set(subject_set, self.output, convertion, window)

      networks = []
      for i in range(0,number_of_classifiers):
         self.config['model_name'] = self.config['model_name'] + "_" + str(i)
         cnn = CNN_MOD_2.CNN_MOD(self.config)
         cnn.set_data_set(self.data_set)
         cnn.train_network()
         networks.append(cnn)
      

      ss_iterator = 0
      num_samples = 0
      while ss_iterator < 10:   
         prediction_steps = 100
         test_set_length = len(self.data_set.test._labels)
         threshold = 0.95
         # Returns an n-long array with random integer
         # integer range, length of array
         test_indecies = np.random.choice(test_set_length, test_set_length, replace=False)


         #print 'Predicting'
         prediction_indices = []
         for i in range(0, len(test_indecies)):
            # Get the data instance
            data = self.data_set.test._data[test_indecies[i]]
            # Predict the class label
            predictions = []
            activities = []
            for j in range(0, number_of_classifiers):
               prediction = networks[j].run_network_return_probability([[data]])[0]
               predictions.append(prediction)
               activity = np.argmax(prediction)
               activities.append(activity)

            max_prediction_index = np.argmax(predictions)/ len(predictions[0])
            if predictions[max_prediction_index][activities[max_prediction_index]] >= threshold:
               prediction_indices.append([test_indecies[i], activities[max_prediction_index]])
            #prediction = self.cnn.run_network_return_probability([[data]])[0]
            #activity = np.argmax(prediction)
            # If the prediction is above a given threshold, add the index to a list
            #if prediction[activity] >= threshold:
            #   prediction_indices.append([test_indecies[i], activity])
         
         print len(prediction_indices)
         num_samples += len(prediction_indices)
         self.data_set = data_generator.move_data_from_test_to_train(prediction_indices, self.data_set)
         for cnn in networks:
            cnn.set_data_set(self.data_set)
            cnn.train_network()
         ss_iterator +=1       

      print 'Number of transfered instances', num_samples
      #self.cnn.save_model('models/' + network_type + '_' + str(input_size) + '_' + str(conv_f_1) + '_' + str(conv_f_2) + '_' + str(nn_1) + '_' + filter_type)


cnn_h = CNN_SS_TRAIN('original', 500 , '1.0', 600, 20, 40, 200, "VALID", 1) 


