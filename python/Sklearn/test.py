import pandas as pd
import VAR 
import numpy as np
import matplotlib.pyplot as plt
import data_features

def main():
	VARIABLES = VAR.VARIABLES()
	keep_activities_dict = VARIABLES.CONVERTION_GROUPED_ACTIVITIES
	model = "RF_1.0_" + str(keep_activities_dict)
	df = load_data(model)
	
	overlap = 5
	length = 100#len(df)

	df_label = df["label"]
	

	df_prediction = df[["prediction","probability"]]
	df_label = df_label[0:length]
	df_prediction = df_prediction[0:length]
	
	df_matrix_prediction = df_prediction.as_matrix()
	print sum(df_matrix_prediction[:,0] == df_label.values) * 1.0 / length
	

	df_changed_threshold = df_prediction.copy()

	#df_changed_threshold["prediction"][df_changed_threshold["probability"] < 0.3] = 0
	
	highest_prediction = get_highest_prediction(df_changed_threshold, length,2,2)

	print len(highest_prediction), len(df_label), "length"
	print sum(highest_prediction[:,0] == df_label[0])

	df_label = return_every_n_rows(df_label, 5)
	df_prediction_old = return_every_n_rows(df_prediction, 5)
	df_label = df_label.reset_index(drop=True)
	df_result = pd.DataFrame(highest_prediction)
	df_result = df_result.reset_index(drop=True)

	#print df_result 
	df_total =  pd.concat([df_label, df_result],axis=1,ignore_index=True)
	
	#print df_total.head()
	#print df_total[0].values
	#print df_total[1].values
	print sum(df_total[1].values == 0)
	print sum(df_total[0].values == df_total[1].values) *1.0 / (length/5)

	start = 0
	stop = 1000
	#graph(start, stop, df_total, df_prediction_old)

	

def graph(start, stop, df_total, df_prediction_old):
	plt.figure(1)

	plt.subplot(511)
	axes = plt.gca()
	axes.set_ylim([0, 11])
	plt.plot(df_total[0][start:stop])
	
	plt.subplot(512)
	axes = plt.gca()
	axes.set_ylim([0, 2])

	plt.plot(df_prediction_old["probability"][start:stop])

	plt.subplot(513)
	axes = plt.gca()
	axes.set_ylim([0, 11])
	plt.plot(df_total[1][start:stop])

	plt.subplot(514)
	axes = plt.gca()
	axes.set_ylim([0,2])
	plt.plot(df_total[2][start:stop])
	#for i in range(0,len(df_total)):
	#	if df_total.iloc[i].values[1] == df_total.iloc[i].values[0]:
	#		print i
	df_true_label = data_features.read_true_label()

	plt.subplot(515)
	axes = plt.gca()
	axes.set_ylim([0, 11])
	plt.plot(df_true_label[0][start*100:stop*100])
	plt.show()

def load_data(model):
	df = pd.read_csv("predictions/" + model + ".csv")
	return df	
	
def get_highest_prediction(df, length, range_down, range_up):
	highest_prediction = np.zeros([length, 2])
	for i in range(0, length):
		if i < 5:
			#print i,'i'
			highest_prediction[i] = get_highest_prediction_probability(df, 0, i, i)
		
		elif i > length-5:
			#print i,'i'
			#print i-range_down, i+(i-length-1)
			highest_prediction[i] = get_highest_prediction_probability(df, range_down,length-i-2, i)

		else:
			#print i
			highest_prediction[i] = get_highest_prediction_probability(df, range_down, range_up, i)
	#print highest_prediction
	return highest_prediction



def get_highest_prediction_probability(df, range_down, range_up, i):
	''' Returns the label and probability for the highest probability within a range '''
	'''
	Input: 
	   prediction  probability
	0           5        0.860
	1         100        0.455
	2           5        0.530
	3         100        0.440
	4         100        0.455
	5           5        0.970

	Output:
	5           5        0.970
	'''

	df_temp = df[i-range_down:i+range_up+1]

	high_prob_index = np.argmax(df_temp["probability"].values)
	high_prob_label = df_temp.iloc[high_prob_index].values
	return high_prob_label

def return_every_n_rows(df, n):
	return df[::n]



if __name__ == "__main__":
    main()