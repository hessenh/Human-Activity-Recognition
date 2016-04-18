import pandas as pd
import numpy as np


overlap = 80
window_size = 100



def load_actual_labels(subjects):
	''' Load actual window labels '''
	path = '../../../Prosjektoppgave/Notebook/data/'+subjects[0]+'/RAW_SIGNALS/'+subjects[0]+'_GoPro_LAB_All.csv'
	df = pd.read_csv(path, header=None, sep='\,',engine='python')

	for i in range(1,len(subjects)):
		path = '../../../Prosjektoppgave/Notebook/data/'+subjects[i]+'/RAW_SIGNALS/'+subjects[i]+'_GoPro_LAB_All.csv'
		df_temp = pd.read_csv(path, header=None, sep='\,',engine='python')
		df = pd.concat([df,df_temp],axis =0, )
	return df

def remove_activities(df):
	# Remove activities
	remove = {12:12, 15:15, 17:17}
	for key, value in remove.iteritems():
	     df =  df[df[0] != key]
	return df

subjects = ['21A', '05A', '14A', '18A', '03A', '22A', '10A']
df_actual = load_actual_labels(subjects)
df_actual = remove_activities(df_actual)
print df_actual.describe()



''' Find shuffling '''
start = 0
stop = 0
start_stop = []
walking = False
for i in range(0, len(df_actual)):
	if df_actual.iloc[i].values[0] == 1.0 and not walking:
		walking = True
		start = i

	elif df_actual.iloc[i].values[0] != 1.0 and walking:
		walking = False
		stop = i
		#print start, stop
		#if stop - start < 15:
		start_stop.append([start, stop])
		#start = i


length = []
for i in range(0,len(start_stop)):
	temp = start_stop[i]
	length.append(temp[1]-temp[0])

print length
