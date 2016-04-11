# Modified Convolutional Neural Network
# 
# ==============================================================================
'''
Change number of conv layers dynamically
Change numbers of neural layers dynamically
Filter: VALID
'''

import CNN_STATIC_VARIABLES
import tensorflow as tf
import numpy as np
import input_data_window_large
import pandas as pd
import time

class CNN_FILTER(object):
  def __init__(self, config):
    self._info = "Convolutional neural network with two convolutional layers and a fully connected network"

    '''Config'''
    self._input_size = config['input_size']
    self._output_size = config['output_size']
    self._iteration_size = config['iteration_size']
    self._batch_size = config['batch_size']
    self._model_name = config['model_name']
  
    filter_x = 30
    filter_y = 1
    resize_y = 6
    resize_x = 100
    window = self._input_size / 6

    kernel_list = [1] + config['conv_list']
    number_of_kernels = len(kernel_list)-1
    connections_in = resize_y * (resize_x - (number_of_kernels*filter_x) + number_of_kernels) * kernel_list[-1]
    neural_list = [connections_in] + config['neural_list'] + [self._output_size]
    FILTER_TYPE = 'VALID'

    
    

    '''Placeholders for input and output'''
    self.x = tf.placeholder("float", shape=[None, self._input_size])
    print self.x.get_shape(),'Input' 
    self.y_ = tf.placeholder("float", shape=[None, self._output_size])
    print self.y_.get_shape(), 'Output'
    self.reshaped_input = tf.reshape(self.x, [-1, resize_y, resize_x, 1])
    print self.reshaped_input.get_shape(), 'Input reshaped'


    def get_conv_layer(input_variables, number_kernels_in, number_kernels_out, layer_number, filter_x, filter_y, filter_type):
      weights = self.weight_variable([filter_y, filter_x, number_kernels_in, number_kernels_out], self._model_name + "w_conv_" + layer_number)
      bias = self.bias_variable([number_kernels_out],self._model_name + 'b_conv_' + layer_number)
      return tf.nn.relu(self.conv2d(input_variables, weights, filter_type) + bias)



    def connect_conv_layers(input_variables):
      
      output = get_conv_layer(input_variables, kernel_list[0], kernel_list[1], '1', filter_x, filter_y, FILTER_TYPE)
      print output.get_shape(), 'Features 1'
      
      for i in range(1,len(kernel_list)-1):
        output = get_conv_layer(output, kernel_list[i], kernel_list[i+1], str(i+1), filter_x, filter_y, FILTER_TYPE)
        print output.get_shape(), 'Features ', i+1

      # Flatten output
      output = tf.reshape(output, [-1, resize_y * (resize_x - (number_of_kernels*filter_x) + number_of_kernels) * kernel_list[-1]])
      return output    
    

    def get_nn_layer(input_variables, connections_in, connections_out, layer_number):
      weights = self.weight_variable([connections_in, connections_out], self._model_name + 'w_fc_' + str(layer_number+1))
      bias = self.bias_variable([connections_out], self._model_name +'b_fc_' + str(layer_number+1))
      return tf.nn.relu(tf.matmul(input_variables, weights) + bias)

    

    def connect_nn_layers(input_variables, keep_prob):
      output = get_nn_layer(input_variables, neural_list[0], neural_list[1],0)
      print output.get_shape(), 'NN',0
      for i in range(1, len(neural_list)-2):
        output = get_nn_layer(output, neural_list[i], neural_list[i+1], i)
        print output.get_shape(),'NN',i


      output = tf.nn.dropout(output, keep_prob)
      # Last layer
      weights = self.weight_variable([neural_list[-2], neural_list[-1]], self._model_name + 'w_fc_last')
      bias = self.bias_variable([neural_list[-1]],self._model_name + 'b_fc_last')
      y_conv = tf.nn.softmax(tf.matmul(output, weights) + bias)
      return y_conv


    ''' Convolutinal layers'''
    self.output_conv = connect_conv_layers(self.reshaped_input)

    self.keep_prob = tf.placeholder("float")
    '''Densly conected layers'''
    self.y_conv = connect_nn_layers(self.output_conv, self.keep_prob)


    self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(tf.clip_by_value(self.y_conv,1e-10,1.0)))
    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
    self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

    self.init_op = tf.initialize_all_variables()

  def weight_variable(self,shape,name):
    #print "weight_variable", shape
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name= name)

  def bias_variable(self, shape, name):
  #print 'bias_variable', shape
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

  def conv2d(self, x, W, filter_type):
    #print 'conv2d', x
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=filter_type)

  def max_pool_2x2(self, x):
  #print 'max_pool_2x2', x
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


  def set_data_set(self, data_set):
    self._data_set = data_set

  def load_model(self, model):
    self.sess = tf.Session()
    # Get last name. Path could be modles/test, so splitting on "/" and retreiving "test"
    model_name = model.split('/')
    model_name = model_name[-1]
    all_vars = tf.all_variables()
    model_vars = [k for k in all_vars if k.name.startswith(model_name)]
    tf.train.Saver(model_vars).restore(self.sess, model)

  def save_model(self, model):
    saver = tf.train.Saver()
    save_path = saver.save(self.sess, model)
    print("Model saved in file: %s" % save_path)

  @property
  def info(self):
    return "Info: " + self._info

  def data_set(self):
    return self._data_set

  ''' Test network with test data set, returning accuracy '''
  def test_network(self):
    return self.sess.run(self.accuracy,feed_dict={
      self.x: self._data_set.test.data, self.y_: self._data_set.test.labels, self.keep_prob: 1.0})

  ''' Return predicted value for a given data instance '''
  def run_network(self, data):
    ''' Predict single data'''
    prediction = self.sess.run(self.y_conv, feed_dict={self.x: data[0],self.keep_prob:1.0})
    prediction = np.argmax(prediction[0])+1
    return prediction

  def run_network_return_probability(self, data):
    prediction = self.sess.run(self.y_conv, feed_dict={self.x: data[0],self.keep_prob:1.0})
    return prediction

  def get_activity_list_accuracy(self, original_data_set, data_set):
    number_of_activities = len(original_data_set.test.labels[0])
    
    activity_accuracy = np.zeros(number_of_activities)
    for i in range(0,number_of_activities):
      pos = original_data_set.test.labels[..., i] == 1
      data = original_data_set.test.data[pos]
      labels = data_set.test.labels[pos]
      #print len(data),i
      #print len(labels),i
      accuracy = self.sess.run(self.accuracy,feed_dict={self.x: data, self.y_: labels, self.keep_prob: 1.0})
      activity_accuracy[i] = accuracy
    return activity_accuracy

  def test_network_stepwise(self):
    length_of_data = len(self._data_set.test.data)
    #print length_of_data
    activities = [0,1,2,3,4,5,6,7,8,9]
    total_accuracy = 0.0
    total_accuracy_whole = 0.0
    for activity in activities:
      #step = length_of_data / 10
      
      activity_boolean = self._data_set.test.labels[::,activity] == 1.0
      activity_data = self._data_set.test.data[activity_boolean]
      activity_label = self._data_set.test.labels[activity_boolean]
      length_of_temp_step = len(activity_data) / 10
      temp_score = 0.0
      for i in range(0, 10):
        temp_data = activity_data[i*length_of_temp_step:i*length_of_temp_step+length_of_temp_step]
        temp_label = activity_label[i*length_of_temp_step:i*length_of_temp_step + length_of_temp_step]
        temp_score += self.sess.run(self.accuracy,feed_dict={
          self.x: temp_data, self.y_: temp_label, self.keep_prob: 1.0})
      accuracy = temp_score / 10
      print str(accuracy).replace(".",",")
      total_accuracy += accuracy
      total_accuracy_whole += accuracy * (len(activity_data)*1.0/length_of_data)

    print 'Accuracy', str(total_accuracy_whole).replace(".",",")
    print 'Sensitivity', str(total_accuracy / len(activities)).replace(".",",")

  ''' Train network '''
  def train_network(self):
    '''Creating model'''
    self.sess = tf.Session()
    self.sess.run(self.init_op)
    for i in range(self._iteration_size):
      batch = self._data_set.train.next_batch(self._batch_size)
      self.sess.run(self.train_step, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
      iteration_list = [1000,1500,3000,4000,5000]
      #if i in iteration_list:
      #  print i, "Test network stepwise"
      #  self.test_network_stepwise()
      #if i%100 == 0:
        #print i, 'Iteration'
        #print(i,self.sess.run(self.accuracy,feed_dict={self.x: self._data_set.test.data, self.y_: self._data_set.test.labels, self.keep_prob: 1.0}))


    #print(self.sess.run(self.accuracy,feed_dict={
     # self.x: self._data_set.test.data, self.y_: self._data_set.test.labels, self.keep_prob: 1.0}))


class CNN_FILTER_WITH_POOLING(object):
  def __init__(self, config):
    self._info = "Convolutional neural network with two convolutional layers and a fully connected network"

    '''Config'''
    self._input_size = config['input_size']
    self._output_size = config['output_size']
    self._iteration_size = config['iteration_size']
    self._batch_size = config['batch_size']
    self._model_name = config['model_name']
  
    filter_x = 35
    filter_y = 1
    resize_y = 6
    resize_x = 100
    window = self._input_size / 6

    kernel_list = [1] + config['conv_list']
    number_of_kernels = len(kernel_list)-1
    connections_in = 2 * 25 * kernel_list[-1]
    neural_list = [connections_in] + config['neural_list'] + [self._output_size]
    FILTER_TYPE = 'VALID'

    
    

    '''Placeholders for input and output'''
    self.x = tf.placeholder("float", shape=[None, self._input_size])
    print self.x.get_shape(),'Input' 
    self.y_ = tf.placeholder("float", shape=[None, self._output_size])
    print self.y_.get_shape(), 'Output'
    self.reshaped_input = tf.reshape(self.x, [-1, resize_y, resize_x, 1])
    print self.reshaped_input.get_shape(), 'Input reshaped'

    def max_pool_2x2(x):
      #print 'max_pool_2x2', x
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_conv_layer(input_variables, number_kernels_in, number_kernels_out, layer_number, filter_x, filter_y, filter_type):
      weights = self.weight_variable([filter_y, filter_x, number_kernels_in, number_kernels_out], self._model_name + "w_conv_" + layer_number)
      bias = self.bias_variable([number_kernels_out],self._model_name + 'b_conv_' + layer_number)
      return tf.nn.relu(self.conv2d(input_variables, weights, filter_type) + bias)



    def connect_conv_layers(input_variables):
      
      output = get_conv_layer(input_variables, kernel_list[0], kernel_list[1], '1', filter_x, filter_y, "SAME")
      # Pooling
      print output.get_shape(), 'Features before pooling 1'
      output = max_pool_2x2(output)
      print output.get_shape(), 'Features 1'
      
      for i in range(1,len(kernel_list)-1):
        output = get_conv_layer(output, kernel_list[i], kernel_list[i+1], str(i+1), filter_x, filter_y, "SAME")
        print output.get_shape(), 'Features before pooling ', i+1
        output = max_pool_2x2(output)
        print output.get_shape(), 'Features ', i+1

      # Flatten output
      output = tf.reshape(output, [-1, 2 * 25 * kernel_list[-1]])
      print output.get_shape(), "Output conv - Reshaped"
      return output    
    

    def get_nn_layer(input_variables, connections_in, connections_out, layer_number):
      weights = self.weight_variable([connections_in, connections_out], self._model_name + 'w_fc_' + str(layer_number+1))
      bias = self.bias_variable([connections_out], self._model_name +'b_fc_' + str(layer_number+1))
      return tf.nn.relu(tf.matmul(input_variables, weights) + bias)

    

    def connect_nn_layers(input_variables, keep_prob):
      output = get_nn_layer(input_variables, neural_list[0], neural_list[1],0)
      print output.get_shape(), 'NN',0
      for i in range(1, len(neural_list)-2):
        output = get_nn_layer(output, neural_list[i], neural_list[i+1], i)
        print output.get_shape(),'NN',i


      output = tf.nn.dropout(output, keep_prob)
      # Last layer
      weights = self.weight_variable([neural_list[-2], neural_list[-1]], self._model_name + 'w_fc_last')
      bias = self.bias_variable([neural_list[-1]],self._model_name + 'b_fc_last')
      y_conv = tf.nn.softmax(tf.matmul(output, weights) + bias)
      return y_conv


    ''' Convolutinal layers'''
    self.output_conv = connect_conv_layers(self.reshaped_input)

    self.keep_prob = tf.placeholder("float")
    '''Densly conected layers'''
    self.y_conv = connect_nn_layers(self.output_conv, self.keep_prob)


    self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(tf.clip_by_value(self.y_conv,1e-10,1.0)))
    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
    self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

    self.init_op = tf.initialize_all_variables()


  def weight_variable(self,shape,name):
    #print "weight_variable", shape
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name= name)

  def bias_variable(self, shape, name):
  #print 'bias_variable', shape
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

  def conv2d(self, x, W, filter_type):
    #print 'conv2d', x
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=filter_type)



  def set_data_set(self, data_set):
    self._data_set = data_set

  def load_model(self, model):
    self.sess = tf.Session()
    # Get last name. Path could be modles/test, so splitting on "/" and retreiving "test"
    model_name = model.split('/')
    model_name = model_name[-1]
    all_vars = tf.all_variables()
    model_vars = [k for k in all_vars if k.name.startswith(model_name)]
    tf.train.Saver(model_vars).restore(self.sess, model)

  def save_model(self, model):
    saver = tf.train.Saver()
    save_path = saver.save(self.sess, model)
    print("Model saved in file: %s" % save_path)

  @property
  def info(self):
    return "Info: " + self._info

  def data_set(self):
    return self._data_set

  ''' Test network with test data set, returning accuracy '''
  def test_network(self):
    return self.sess.run(self.accuracy,feed_dict={
      self.x: self._data_set.test.data, self.y_: self._data_set.test.labels, self.keep_prob: 1.0})

  ''' Return predicted value for a given data instance '''
  def run_network(self, data):
    ''' Predict single data'''
    prediction = self.sess.run(self.y_conv, feed_dict={self.x: data[0],self.keep_prob:1.0})
    prediction = np.argmax(prediction[0])+1
    return prediction

  def run_network_return_probability(self, data):
    prediction = self.sess.run(self.y_conv, feed_dict={self.x: data[0],self.keep_prob:1.0})
    return prediction

  def get_activity_list_accuracy(self, original_data_set, data_set):
    number_of_activities = len(original_data_set.test.labels[0])
    
    activity_accuracy = np.zeros(number_of_activities)
    for i in range(0,number_of_activities):
      pos = original_data_set.test.labels[..., i] == 1
      data = original_data_set.test.data[pos]
      labels = data_set.test.labels[pos]
      #print len(data),i
      #print len(labels),i
      accuracy = self.sess.run(self.accuracy,feed_dict={self.x: data, self.y_: labels, self.keep_prob: 1.0})
      activity_accuracy[i] = accuracy
    return activity_accuracy

  def test_network_stepwise(self):
    length_of_data = len(self._data_set.test.data)
    #print length_of_data
    activities = [0,1,2,3,4,5,6,7,8,9]
    total_accuracy = 0.0

    for activity in activities:
      #step = length_of_data / 10
      
      activity_boolean = self._data_set.test.labels[::,activity] == 1.0
      activity_data = self._data_set.test.data[activity_boolean]
      activity_label = self._data_set.test.labels[activity_boolean]
      accuracy = self.sess.run(self.accuracy,feed_dict={
        self.x: activity_data, self.y_: activity_label, self.keep_prob: 1.0})
      print activity+1, accuracy
      total_accuracy += accuracy
    return total_accuracy / len(activities)

  ''' Train network '''
  def train_network(self):
    '''Creating model'''
    self.sess = tf.Session()
    self.sess.run(self.init_op)
    for i in range(self._iteration_size):
      batch = self._data_set.train.next_batch(self._batch_size)
      self.sess.run(self.train_step, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
        #print(i,self.sess.run(self.accuracy,feed_dict={self.x: self._data_set.test.data, self.y_: self._data_set.test.labels, self.keep_prob: 1.0}))
    #print(self.sess.run(self.accuracy,feed_dict={
     # self.x: self._data_set.test.data, self.y_: self._data_set.test.labels, self.keep_prob: 1.0}))


if __name__ == "__main__":
  test = True
  VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
  subject_set = VARS.get_subject_set(False)  
  remove_activities = VARS.CONVERTION_STATIC_DYNAMIC_INVERSE
  keep_activities = VARS.CONVERTION_STATIC_DYNAMIC
  output = 10
  config = VARS.get_config(600, output, 20000, 100, 'sd',[20,40], [1500], "VALID")
  

  if test:
    data_set = input_data_window_large.read_data_sets_without_activity([['01A'],subject_set[1]], output, remove_activities, None, keep_activities, "1.0")
    model = config['model_name']
    print model
    cnn = CNN_FILTER(config)
    cnn.set_data_set(data_set)
    cnn.load_model('models/' + model)
    cnn.test_network_stepwise()

  else:
    data_set = input_data_window_large.read_data_sets_without_activity(subject_set, output, remove_activities, None, keep_activities, "1.0")
    data_set.train.shuffle_data_set()
    model = config['model_name']
    cnn = CNN_FILTER(config)
    cnn.set_data_set(data_set)
    cnn.train_network()
    cnn.save_model('models/' + model)
    #cnn.test_network_stepwise()

      