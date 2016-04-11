"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import pandas as pd
import numpy as np

def extract_merged_axes(subject, window):
  filepath = '../../../../Prosjektoppgave/Notebook/data/'+subject+'/DATA_WINDOW/'+window+'/ORIGINAL/'
  files =   [
  'Axivity_BACK_Back_X.csv', 'Axivity_THIGH_Right_Y.csv', 
  'Axivity_BACK_Back_Y.csv', 'Axivity_THIGH_Right_Z.csv', 
  'Axivity_BACK_Back_Z.csv', 'Axivity_THIGH_Right_X.csv']
  df_0 = pd.read_csv(filepath+files[0], header=None, sep='\,',engine='python')
  df_1 = pd.read_csv(filepath+files[1], header=None, sep='\,',engine='python')
  df_2 = pd.read_csv(filepath+files[2], header=None, sep='\,',engine='python')
  df_3 = pd.read_csv(filepath+files[3], header=None, sep='\,',engine='python')
  df_4 = pd.read_csv(filepath+files[4], header=None, sep='\,',engine='python')
  df_5 = pd.read_csv(filepath+files[5], header=None, sep='\,',engine='python')
  df = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5],axis=1)
  return df.as_matrix(columns=None)

def extract_data(subjects, window):
  print('Extracting data from', subjects)

  train_data = extract_merged_axes(subjects[0], window)

  for i in range(1,len(subjects)):
    subject_data = extract_merged_axes(subjects[i], window)

    train_data = np.concatenate((train_data,subject_data ), axis=0)

  return train_data

def extract_merged_labels(subject, output_size, change_labels, window):
  filepath = '../../../../Prosjektoppgave/Notebook/data/'+subject+'/DATA_WINDOW/'+window+'/ORIGINAL/GoPro_LAB_All_L.csv'
  

  df = pd.read_csv(filepath, header=None, sep='\,',engine='python')

  # Convert from nunber to readable format: 
  # From: [2]
  # To: [0,1,0,0,0,0,0,0,0,0,0]
  m = []
  for i in range(len(df)):
    a = df.iloc[i]
    if change_labels:
      a = change_labels[a.values[0]]
    n =  np.zeros(output_size)
    n[a-1] = 1
    m.append(n)

  s = pd.DataFrame(m)


  return s.values


def extract_labels(subjects, output_size, change_labels, window):
  print('Extracting label from', subjects)
  train_label = extract_merged_labels(subjects[0], output_size, change_labels, window)

  for i in range(1,len(subjects)):
    subject_label = extract_merged_labels(subjects[i], output_size, change_labels, window)

    train_label = np.concatenate((train_label,subject_label ), axis=0)

  return train_label


def extract_merged_labels_and_data(subject, output_size, remove_activities, convert_activties, window):
  filepath = '../../../Prosjektoppgave/Notebook/data/'+subject+'/DATA_WINDOW/'+window+'/ORIGINAL/'
  files =   [
  'Axivity_BACK_Back_X.csv', 'Axivity_THIGH_Right_Y.csv', 
  'Axivity_BACK_Back_Y.csv', 'Axivity_THIGH_Right_Z.csv', 
  'Axivity_BACK_Back_Z.csv', 'Axivity_THIGH_Right_X.csv']
  df_0 = pd.read_csv(filepath+files[0], header=None, sep='\,',engine='python')
  df_1 = pd.read_csv(filepath+files[1], header=None, sep='\,',engine='python')
  df_2 = pd.read_csv(filepath+files[2], header=None, sep='\,',engine='python')
  df_3 = pd.read_csv(filepath+files[3], header=None, sep='\,',engine='python')
  df_4 = pd.read_csv(filepath+files[4], header=None, sep='\,',engine='python')
  df_5 = pd.read_csv(filepath+files[5], header=None, sep='\,',engine='python')

  filepath = '../../../Prosjektoppgave/Notebook/data/'+subject+'/DATA_WINDOW/'+window+'/ORIGINAL/GoPro_LAB_All_L.csv'

  df_labels = pd.read_csv(filepath, header=None, sep='\,',engine='python')
  df_labels.columns = ['labels']
  df_data = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_labels],axis=1)

  for key, value in remove_activities.iteritems():
     df_data =  df_data[df_data['labels'] != key]
  df_labels = df_data['labels']
  df_data = df_data.drop('labels', 1)

  # Convert from nunber to readable format: 
  # From: [2]
  # To: [0,1,0,0,0,0,0,0,0,0,0]
  m = []
  for i in range(len(df_labels)):
    a = df_labels.iloc[i]
    n =  np.zeros(output_size)
    #print(a,n,convert_activties)
    n[convert_activties.get(a)-1] = 1
    m.append(n)

  df_labels = pd.DataFrame(m)

  return df_data.as_matrix(columns=None), df_labels.values

def extract_labels_and_data(subjects, output_size, remove_activities, convert_activties, window):
  print('Extracting label and data set from', subjects)
  data, labels = extract_merged_labels_and_data(subjects[0], output_size, remove_activities, convert_activties, window)

  # Iterate over all subjects
  for i in range(1,len(subjects)):
    sub_data, sub_labels = extract_merged_labels_and_data(subjects[i], output_size, remove_activities, convert_activties, window)

    # Append data and labels
    data = np.concatenate((data,sub_data ), axis=0)
    labels = np.concatenate((labels, sub_labels), axis=0)

  return data, labels

class DataSet(object):

  def __init__(self, data, labels):
    self._num_examples = data.shape[0]

    self._data = data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  def label_size(self):
      return len(self._labels)
  

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed


  def shuffle_data_set(self):
    perm = numpy.arange(len(self._data))
    numpy.random.shuffle(perm)
    self._data = self._data[perm]
    self._labels = self._labels[perm]
    #print('Data set is shuffled')
    
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
   
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._data = self._data[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._data[start:end], self._labels[start:end]

  def next_data_label(self, index):
    """Return the next `batch` examples from this data set."""
    if index > len(self._data):
      print('Test index is larger than data set', index)
      index = 0 

    return [self._data[index]], self._labels[index]


 

def read_data_sets(subjects_set, output_size, change_labels, load_model, window):
  training_subjects = subjects_set[0]
  test_subjects = subjects_set[1]
  
  class DataSets(object):
    pass
  data_sets = DataSets()

  # If the model is for testing
  if load_model:
    # Testing data and labels
    test_data = extract_data(test_subjects, window)
    test_labels = extract_labels(test_subjects, output_size, change_labels, window)

    # Define testing data sets
    data_sets.test = DataSet(test_data, test_labels)

  else:
     # Training data and labels
    train_data = extract_data(training_subjects, window)
    train_labels = extract_labels(training_subjects, output_size, change_labels, window)

    # Testing data and labels
    test_data = extract_data(test_subjects,window)
    test_labels = extract_labels(test_subjects, output_size, change_labels, window)

    # Define training and testing data sets
    data_sets.train = DataSet(train_data, train_labels)
    data_sets.test = DataSet(test_data, test_labels)

  return data_sets


def read_data_sets_without_activity(subjects_set, output_size, remove_activities, load_model, convert_activties, window):
  training_subjects = subjects_set[0]
  test_subjects = subjects_set[1]
  
  class DataSets(object):
    pass
  data_sets = DataSets()

  # If the model is for testing
  if load_model:
    # Testing data and labels
    test_data, test_labels = extract_labels_and_data(test_subjects, output_size, remove_activities, convert_activties, window)

    # Define testing data sets
    data_sets.test = DataSet(test_data, test_labels)

  else:
     # Training data and labels
    train_data, train_labels = extract_labels_and_data(training_subjects, output_size, remove_activities, convert_activties, window)
    oversampling = False;
    if oversampling:
      # length of longest activity
      max_length = 0
      activities = [0,1,2,3,4,5,6,7,8,9]
      for activity in activities:
        activity_length = sum(train_labels[::,activity])
        if activity_length > max_length:
          max_length = activity_length
      print(max_length)
      train_data_new = np.zeros([max_length * len(activities), 600])
      train_labels_new = np.zeros([max_length * len(activities), len(activities)])
      #print(train_labels[0:10], 'old')
      #print(train_labels_new[0:10], 'new')

      for i in range(0,len(activities)):
        activity_boolean = train_labels[::,i] == 1.0
        activity_data = train_data[activity_boolean]
        activity_label = train_labels[activity_boolean]

        activity_length = len(activity_data)
        fraction = int(max_length / activity_length) + 1
        
        new_activity_data = np.tile(activity_data, (fraction, 1))
        new_activity_label = np.tile(activity_label, (fraction, 1))
        new_activity_data = new_activity_data[0:max_length]
        new_activity_label = new_activity_label[0:max_length]
        train_data_new[i*max_length:i*max_length+max_length] = new_activity_data
        train_labels_new[i*max_length:i*max_length+max_length] = new_activity_label

      for activity in activities:
        activity_length = sum(train_labels_new[::,activity])
        print(activity_length)
      # Testing data and labels
      train_data = train_data_new
      train_labels = train_labels_new
    test_data, test_labels = extract_labels_and_data(test_subjects, output_size, remove_activities, convert_activties, window)
    # Define training and testing data sets
    data_sets.train = DataSet(train_data, train_labels)
    data_sets.test = DataSet(test_data, test_labels)

  return data_sets

