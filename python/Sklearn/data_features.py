''' 
Data: Features and labels
Data set object with training and test data. 
Subjects is collected from VAR file

variables:
 - train_x
 - train_l
 - test_x
 - test_l
 Variables have format [[data, data,...,data],[data,data,..,data]] and [[label],[label]]
'''
import pandas as pd
import numpy as np
import VAR

class Data_Set(object):
  """docstring for Data_set"""
  def __init__(self, remove_activities):
    VARIABLES = VAR.VARIABLES()
    window = '1.0'

    train = read_subjects(VARIABLES.TRAIN_SUBJECTS, window)
    self.train_x = train[0]
    self.train_l = train[1]
    print len(self.train_l)
    print len(self.train_x)
    remove_activities = [6]
    for activitiy in remove_activities:
      pos = self.train_l != activitiy
      self.train_l = np.delete(self.train_l, pos)
      self.train_x = np.delete(self.train_x, [True, False])
    print self.train_x
    print len(self.train_l)
    print len(self.train_x)
    test = read_subjects(VARIABLES.TEST_SUBJECTS, window)
    self.test_x = test[0]
    self.test_l = test[1]

   

def read_subjects(subjects, window):
  data_l = read_set_label(subjects[0], window)
  # Assures that the length of data is the same as the length of labels
  length = len(data_l)
  data_x = read_set_x(subjects[0], window, length)

  for i in range(1,len(subjects)):
    subject_l = read_set_label(subjects[i],window)
      # Assures that the length of data is the same as the length of labels
    length = len(subject_l)
    subject_x = read_set_x(subjects[i], window, length)
    
    data_x = np.concatenate((data_x,subject_x), axis=0)
    data_l = np.concatenate((data_l,subject_l), axis=0)

  return data_x, data_l

def read_set_x(subject, window, length):
  filepath = '../../../Prosjektoppgave/Notebook/data/'+subject+'/FEATURES/'+window+'/FEATURES.csv'
 
  df = pd.read_csv(filepath, sep='\,',engine='python')

  return df.as_matrix(columns=None)[:length]

def read_set_label(subject, window):
  filepath = '../../../Prosjektoppgave/Notebook/data/'+subject+'/DATA_WINDOW/'+window+'/ORIGINAL/GoPro_LAB_All_L.csv'
 
  df = pd.read_csv(filepath, sep='\,',engine='python')

  return df.as_matrix(columns=None)

