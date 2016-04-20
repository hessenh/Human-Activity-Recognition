import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


''' Load viterbi output '''
def load_viterbi_result(start, stop):
	viterbi_path = 'viterbi_sd.csv'
	df_viterbi = pd.read_csv(viterbi_path, header=None, sep='\,',engine='python')
	return df_viterbi[start:stop]


def load_actual_labels(subjects, start, stop):
	''' Load actual window labels '''
	path = '../../../Prosjektoppgave/Notebook/data/'+subjects[0]+'/DATA_WINDOW/1.0/ORIGINAL/GoPro_LAB_All_L.csv'
	df = pd.read_csv(path, header=None, sep='\,',engine='python')

	for i in range(1,len(subjects)):
		path = '../../../Prosjektoppgave/Notebook/data/'+subjects[i]+'/DATA_WINDOW/1.0/ORIGINAL/GoPro_LAB_All_L.csv'
		df_temp = pd.read_csv(path, header=None, sep='\,',engine='python')
		df = pd.concat([df,df_temp],axis =0, )
	df = remove_activities(df)
	df = convert_activities(df)

	return df[start:stop]


def remove_activities(df):
	# Remove activities
	remove = {12:12, 15:15, 17:17}
	for key, value in remove.iteritems():
	     df =  df[df[0] != key]
	return df


def convert_activities(df):
	df_new = df.copy(deep=True)
	# Convert activities
	convertion = {1:1, 2:2, 3:11, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8, 11:8, 13:9, 14:10, 16:12, 9:13}
	for key, value in convertion.iteritems():
		df_new[df[0] == key] = value
	return df_new



def find_shuffling(df):
	''' Find shuffling '''
	start = 0
	stop = 0
	start_stop = []
	walking = False
	for i in range(0, len(df)):
		if df.iloc[i].values[0] == 1.0 and not walking:
			walking = True
			start = i

		elif df.iloc[i].values[0] != 1.0 and walking:
			walking = False
			stop = i
			#print start, stop
			if stop - start < 10:
				start_stop.append([start, stop])
			#start = i
	for s in start_stop:
		start = s[0]
		stop = s[1]
		for i in range(start, stop):
			df.iloc[i] = 11

	return df

def get_score(df, df_actual, activities):
	score = 0

	activity_accuracy=np.zeros(len(activities))
	for i in range(0,len(df)):
		viterbi = df.iloc[i].values[0]
		actual = df_actual.iloc[i].values[0]
		if viterbi == actual:
			activity_accuracy[viterbi-1] +=1.0

	# Divide on number of activities
	for activity in activities:
		length = len(df_actual[df_actual[0]==activity])
		activity_accuracy[activity-1] = activity_accuracy[activity-1]/length

	return activity_accuracy




def visualize(df_list):
	plt.figure(1)
	#y_values = ["W", "R", "S-D", "S-U", "ST", "SI", "L", "P", "C-S", "C-ST", "SH", "V", "T"]
	#y_axis = np.arange(1,14,1)


	plt.subplot(311)
	axes = plt.gca()
	axes.set_ylim([0.9,13.4])
	#plt.yticks(y_axis, y_values)
	plt.plot(df_list[0])

	plt.subplot(312)
	axes = plt.gca()
	axes.set_ylim([0.9,13.4])
	#plt.yticks(y_axis, y_values)
	plt.plot(df_list[1])



	plt.show()


def main():
	start= 2500
	stop = 3000
	subjects = ['21A', '05A', '14A', '18A', '03A', '22A', '10A']
	df = load_viterbi_result(start,stop)
	df_viterbi = find_shuffling(df)
	df_actual = load_actual_labels(subjects, start, stop)
	activities = [1,2,3,4,5,6,7,8,9,10, 11, 12, 13]
	print get_score(df_viterbi, df_actual, activities )
	
	visualize([df_actual, df_viterbi])

main()