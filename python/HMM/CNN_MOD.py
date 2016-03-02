# Modified Convolutional Neural Network
# 
# ==============================================================================

import tensorflow as tf
import numpy as np
import input_data_window_large
import pandas as pd
import time
import CNN_STATIC_VARIABLES

class CNN_MOD(object):
  def __init__(self, config):
    self._info = "Convolutional neural network with two convolutional layers and a fully connected network"

    '''Config'''
    self._input_size = config['input_size']
    self._output_size = config['output_size']
    self._iteration_size = config['iteration_size']
    self._batch_size = config['batch_size']
    self._model_name = config['model_name']

    self._w_b_c_1 = config['conv_f_1']
    self._w_b_c_2 = config['conv_f_2']
    self._w_b_n_1 = config['nn_1']
    FILTER_TYPE = config['filter_type']

    filter_x = 6
    filter_y = 6
    window = self._input_size / filter_y

    '''Placeholders for input and output'''
    self.x = tf.placeholder("float", shape=[None, self._input_size])
    print 'x', self.x.get_shape()
    self.y_ = tf.placeholder("float", shape=[None, self._output_size])

    self.x_image = tf.reshape(self.x, [-1, filter_y, window, 1])
    print self.x_image.get_shape(), 'X reshaped'
    '''First convolutional layer'''
    self.W_conv1 = self.weight_variable([filter_x, filter_y, 1, self._w_b_c_1], self._model_name + "W_conv1")
    self.b_conv1 = self.bias_variable([self._w_b_c_1],self._model_name + 'b_conv1')
    self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1, FILTER_TYPE) + self.b_conv1)
    print self.h_conv1.get_shape(), 'Features 1'
    
    '''Second convolutional layer'''
    if FILTER_TYPE == "SAME":
      self.W_conv2 = self.weight_variable([filter_x, filter_y, self._w_b_c_1, self._w_b_c_2], self._model_name + 'W_conv2')
    else:
      self.W_conv2 = self.weight_variable([1, filter_y, self._w_b_c_1, self._w_b_c_2], self._model_name + 'W_conv2')

    self.b_conv2 = self.bias_variable([self._w_b_c_2], self._model_name +'b_conv2')
    self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.W_conv2, FILTER_TYPE) + self.b_conv2)
    print self.h_conv2.get_shape(), 'Features 2'

    '''Densly conected layer'''
    if FILTER_TYPE == "SAME":
      self.h_flat = tf.reshape(self.h_conv2, [-1, filter_y * window * self._w_b_c_2])
      self.W_fc1 = self.weight_variable([filter_y * window * self._w_b_c_2, self._w_b_n_1],self._model_name + 'W_fc1')
    else:  
      self.h_flat = tf.reshape(self.h_conv2, [-1, 1 * (window-5-5) * self._w_b_c_2])
      self.W_fc1 = self.weight_variable([1 * (window-5-5) * self._w_b_c_2, self._w_b_n_1],self._model_name + 'W_fc1')
    print self.h_flat.get_shape(), 'Output conv'
    print self.W_fc1.get_shape(), 'Neural network input'
    self.b_fc1 = self.bias_variable([self._w_b_n_1], self._model_name +'b_fc1')
    self.h_fc1 = tf.nn.relu(tf.matmul(self.h_flat, self.W_fc1) + self.b_fc1)
    self.keep_prob = tf.placeholder("float")
    self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
    self.W_fc2 = self.weight_variable([self._w_b_n_1, self._output_size],self._model_name + 'W_fc2')
    print self.W_fc2.get_shape()
    self.b_fc2 = self.bias_variable([self._output_size],self._model_name + 'b_fc2')
    print self.b_fc2.get_shape()
    self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

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

  ''' Train network '''
  def train_network(self):
    '''Creating model'''
    self.sess = tf.Session()
    self.sess.run(self.init_op)
    for i in range(self._iteration_size):
      batch = self._data_set.train.next_batch(self._batch_size)
      
      self.sess.run(self.train_step, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
      if i%100 == 0:
        print i
      	#print(i,self.sess.run(self.accuracy,feed_dict={self.x: self._data_set.test.data, self.y_: self._data_set.test.labels, self.keep_prob: 1.0}))


    print(self.sess.run(self.accuracy,feed_dict={
      self.x: self._data_set.test.data, self.y_: self._data_set.test.labels, self.keep_prob: 1.0}))
