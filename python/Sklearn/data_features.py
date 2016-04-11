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
  def __init__(self, keep_activities, remove_activities,keep_activities_original_dict, remove_activities_original_dict, testing):
    VARIABLES = VAR.VARIABLES()
    window = '1.0'
    #keep_activities = VARIABLES.CONVERTION_GROUPED_ACTIVITIES
    #remove_activities = VARIABLES.CONVERTION_GROUPED_ACTIVITIES_INVERSE
    #print "Loading training data"
    train = read_subjects(VARIABLES.TRAIN_SUBJECTS, window, keep_activities, remove_activities, testing)
    self.train_x = train[0]
    self.train_l = train[1]
    shuffled_data_set = shuffle_data_set(self.train_x, self.train_l)
    self.train_x = shuffled_data_set[0]
    self.train_l = shuffled_data_set[1]
    
    #print "Loading test data"
    test = read_subjects(VARIABLES.TEST_SUBJECTS, window, keep_activities, remove_activities, testing)
    self.test_x = test[0]
    self.test_l = test[1]

    #print "Loading original data"
    #keep_activities = VARIABLES.CONVERTION_ORIGINAL
    #remove_activities = VARIABLES.CONVERTION_ORIGINAL_INVERSE
    #test_original = read_subjects(VARIABLES.TEST_SUBJECTS, window, keep_activities, remove_activities)
    test_original = read_subjects(VARIABLES.TEST_SUBJECTS, window, keep_activities_original_dict, remove_activities_original_dict, testing)
    self.test_original_x = test_original[0]
    self.test_original_l = test_original[1]


def shuffle_data_set(data, labels):
  perm = np.arange(len(data))
  np.random.shuffle(perm)
  data = data.as_matrix()[perm]
  labels = labels.as_matrix()[perm]
  data = pd.DataFrame(data)
  labels = pd.DataFrame(labels)
  return data, labels


def read_subjects(subjects, window,keep_activities, remove_activities, testing):
  data_l = read_set_label(subjects[0], window, testing)
  # Assures that the length of data is the same as the length of labels
  length = len(data_l)
  data_x = read_set_x(subjects[0], window, length, testing)

  for i in range(1,len(subjects)):
    subject_l = read_set_label(subjects[i],window, testing)
    # Assures that the length of data is the same as the length of labels
    length = len(subject_l)
    subject_x = read_set_x(subjects[i], window, length, testing)
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


def read_true_label():
  VARIABLES = VAR.VARIABLES()
  subjects = VARIABLES.TEST_SUBJECTS

  filepath = '../../../Prosjektoppgave/Notebook/data/'+subjects[0]+'/RAW_SIGNALS/'+subjects[0]+'_GoPro_LAB_All.csv'
  data = pd.read_csv(filepath, sep='\,', header=None,engine='python')

  for i in range(1,len(subjects)):
    filepath = filepath = '../../../Prosjektoppgave/Notebook/data/'+subjects[i]+'/RAW_SIGNALS/'+subjects[i]+'_GoPro_LAB_All.csv'
    data = pd.concat([data,pd.read_csv(filepath, sep='\,',header=None,engine='python')],axis =0, )
  # Remove activities
  for activity in VARIABLES.CONVERTION_GROUPED_ACTIVITIES_INVERSE:
    data = data[data[0] != activity]

  # Convert activities
  data_l_new = data.copy(deep=True)
  for activities in VARIABLES.CONVERTION_GROUPED_ACTIVITIES:
    data_l_new.ix[data[0] == activities] = VARIABLES.CONVERTION_GROUPED_ACTIVITIES[activities]
  return data_l_new

def read_set_x(subject, window, length, testing):
  if testing:
    filepath = '../../../Prosjektoppgave/Notebook/data/'+subject+'/FEATURES/'+window+'/FEATURES.csv'
  else:
    filepath = '../../../Prosjektoppgave/Notebook/data/'+subject+'/FEATURES/'+window+'/FEATURES_REMOVED_MESSY_WINDOWS.csv'

  df = pd.read_csv(filepath, sep='\,',engine='python')
  return df[0:length]
  #return df.as_matrix(columns=None)[:length]

def read_set_label(subject, window, testing):
  if testing:
    filepath = '../../../Prosjektoppgave/Notebook/data/'+subject+'/DATA_WINDOW/'+window+'/ORIGINAL/GoPro_LAB_All_L.csv'
  else:
    filepath = '../../../Prosjektoppgave/Notebook/data/'+subject+'/DATA_WINDOW/'+window+'/ORIGINAL/GoPro_LAB_All_L_REMOVED_MESSY_WINDOWS.csv'

  df = pd.read_csv(filepath, sep='\,', header=None ,engine='python')

  return df
  #return df.as_matrix(columns=None)

