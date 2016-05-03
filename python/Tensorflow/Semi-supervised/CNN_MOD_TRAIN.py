import data_generator
import CNN_STATIC_VARIABLES
import CNN_MOD_4
import numpy as np

class CNN_SS_TRAIN(object):
   """docstring for CNN_H"""
   def __init__(self, network_type, iterations, window, input_size, conv, nn, filter_type, number_of_classifiers):
      VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
      subject_set = VARS.get_subject_set_SS(False)
      
      output = 10
      remove_activities = VARS.CONVERTION_STATIC_DYNAMIC_INVERSE
      keep_activities = VARS.CONVERTION_STATIC_DYNAMIC

      self.config = VARS.get_config(input_size, output, iterations, 100, network_type, conv, nn, filter_type)
      print 'Creating data set'
      self.data_set = data_generator.read_data_sets_without_activity(subject_set, output, remove_activities, None, keep_activities, window)

      networks = []
      for i in range(0,number_of_classifiers):
         #self.config['model_name'] = self.config['model_name'] + "_" + str(i)
         cnn = CNN_MOD_4.CNN_FILTER(self.config)
         cnn.load_model('models/'+self.config['model_name'])
         cnn.set_data_set(self.data_set)
         #cnn.train_network()
         cnn.test_network_stepwise()
         networks.append(cnn)

      ss_iterator = 0
      num_samples = 0
      while ss_iterator < 3:   
         prediction_steps = 10
         test_set_length = len(self.data_set.test._labels)
         threshold = 0.99
         # Returns an n-long array with random integer
         # integer range, length of array
         pool_size = 1000
         test_indecies = np.random.choice(test_set_length, pool_size, replace=False)

         number_of_samples = 40
         #print 'Predicting'
         prediction_indices = np.zeros([pool_size, 3])
         final_prediction_indices = np.zeros([number_of_samples*10,3])
         for i in range(0, len(test_indecies)):
            # Get the data instance
            data = self.data_set.test._data[test_indecies[i]]
            # Predict the class label
            predictions = np.zeros((number_of_classifiers, output))    
            # Add the different prediction vectors to a list
            for j in range(0, number_of_classifiers):
               prediction = networks[j].run_network_return_probability([[data]])[0]
               predictions[j] = prediction

            activity = np.argmax(predictions)
            confidens = prediction[activity]

            prediction_indices[i] = [test_indecies[i], activity, confidens]

         final_prediction_indices = []
         equal_pool_size = False
         threshold_subset = False
         highest_confident = True
         self_learning = False
         if equal_pool_size:
            ''' Select the N most confident samples from each class '''
            activity_list = [0,1,2,3,4,5,6,7]
            # Sort
            prediction_indices = prediction_indices[prediction_indices[:,2].argsort()]
            for i in range(0, len(activity_list)):
               #print prediction_indices[prediction_indices[:,1] == activity]
               top_predictions = prediction_indices[prediction_indices[:,1] == activity_list[i]][-number_of_samples:]
               for item in top_predictions:
                  final_prediction_indices.append(item)
         if threshold_subset:
            ''' Select random subsample with confidens over threshold'''
            prediction_indices = prediction_indices[prediction_indices[:,2] >= 0.8]
            prediction_indices = prediction_indices[prediction_indices[:,2] >= 0.9]
            print 'Number of prediction instances', len(prediction_indices)
            subset = np.random.choice(len(prediction_indices), number_of_samples*10, replace=False)
            final_prediction_indices = prediction_indices[subset]

         if highest_confident:
            ''' Select the top n samples'''
            # Sort
            prediction_indices = prediction_indices[prediction_indices[:,2].argsort()]
            final_prediction_indices = prediction_indices[-number_of_samples*10:]
            print final_prediction_indices[0]
         if self_learning:
            prediction_indices = prediction_indices[prediction_indices[:,2].argsort()]
            final_prediction_indices = prediction_indices[:number_of_samples*10]


         print 'Number of new instances',len(final_prediction_indices)
         num_samples += len(final_prediction_indices)
         self.data_set = data_generator.move_data_from_test_to_train(final_prediction_indices, self.data_set)
        
         activity_accuracy = np.zeros(len(self.data_set.validation.labels[0]))
         for cnn in networks:
            cnn.set_data_set(self.data_set)
            train_iterations = 800

            print 'Number of training iterations', train_iterations
            cnn.continue_training(train_iterations)
            cnn.test_network_stepwise()
         ss_iterator +=1       

      print 'Number of transfered instances', num_samples
      #self.cnn.save_model('models/' + network_type + '_' + str(input_size) + '_' + str(conv_f_1) + '_' + str(conv_f_2) + '_' + str(nn_1) + '_' + filter_type)


cnn_h = CNN_SS_TRAIN('sd', 20000 , '1.0', 600, [20, 40], [1500], "VALID", 1) 
   

