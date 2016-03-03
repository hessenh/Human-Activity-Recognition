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
  def __init__(self):
    VARIABLES = VAR.VARIABLES()
    window = '1.0'
    keep_activities = VARIABLES.CONVERTION_GROUPED_ACTIVITIES
    remove_activities = VARIABLES.CONVERTION_GROUPED_ACTIVITIES_INVERSE
    print "Loading training data"
    train = read_subjects(VARIABLES.TRAIN_SUBJECTS, window, keep_activities, remove_activities)
    self.train_x = train[0]
    self.train_l = train[1]
    
    print "Loading test data"
    test = read_subjects(VARIABLES.TEST_SUBJECTS, window, keep_activities, remove_activities)
    self.test_x = test[0]
    self.test_l = test[1]

    print "Loading original data"
    keep_activities = VARIABLES.CONVERTION_ORIGINAL
    remove_activities = VARIABLES.CONVERTION_ORIGINAL_INVERSE
    test_original = read_subjects(VARIABLES.TEST_SUBJECTS, window, keep_activities, remove_activities)
    self.test_original_x = test_original[0]
    self.test_original_l = test_original[1]


def read_subjects(subjects, window,keep_activities, remove_activities):
  data_l = read_set_label(subjects[0], window)
  # Assures that the length of data is the same as the length of labels
  length = len(data_l)
  data_x = read_set_x(subjects[0], window, length)

  for i in range(1,len(subjects)):
    subject_l = read_set_label(subjects[i],window)
      # Assures that the length of data is the same as the length of labels
    length = len(subject_l)
    subject_x = read_set_x(subjects[i], window, length)
    
    data_x = pd.concat([data_x,subject_x],axis =0)
    data_l = pd.concat([data_l,subject_l], axis= 0)

  # Remove activities
  for activity in remove_activities:
    data_x = data_x[data_l[0] != activity]
    data_l = data_l[data_l[0] != activity]

  # Convert activities
  data_l_new = data_l.copy(deep=True)
  for activities in keep_activities:
    data_l_new.ix[data_l[0] == activities] = keep_activities[activities]

  return data_x, data_l_new

def read_set_x(subject, window, length):
  filepath = '../../../Prosjektoppgave/Notebook/data/'+subject+'/FEATURES/'+window+'/FEATURES.csv'
 
  df = pd.read_csv(filepath, sep='\,',engine='python')
  return df[0:length]
  #return df.as_matrix(columns=None)[:length]

def read_set_label(subject, window):
  filepath = '../../../Prosjektoppgave/Notebook/data/'+subject+'/DATA_WINDOW/'+window+'/ORIGINAL/GoPro_LAB_All_L.csv'
 
  df = pd.read_csv(filepath, sep='\,', header=None ,engine='python')

  return df
  #return df.as_matrix(columns=None)

