import input_data_window_large
import CNN_MOD_4
import CNN_STATIC_VARIABLES
import numpy as np
import pandas as pd

def get_raw_signal_labels(subject_set):
	print "Inni get raw signal labels"
	filepath = '../../../../Prosjektoppgave/Notebook/data/'+subject_set[1][0]+'/RAW_SIGNALS/'+subject_set[1][0]+'_GoPro_LAB_All.csv'
	df = pd.read_csv(filepath, header=None, sep='\,',engine='python')

	for i in range(1,len(subject_set[1])):
		filepath = '../../../../Prosjektoppgave/Notebook/data/'+subject_set[1][i]+'/RAW_SIGNALS/'+subject_set[1][i]+'_GoPro_LAB_All.csv'
		df_new= pd.read_csv(filepath,header=None, sep='\,',engine='python')
		df = pd.concat([df, df_new ], ignore_index=True)

	#m = np.zeros(len(df))
	#for i in range(len(df)):
	#	a = df.iloc[i]
	#	a = change_labels[a.values[0]]
	#	m[i]=a
	#print len(m)
	#print m
	return df   

class CNN_TEST(object):
	"""docstring for CNN_H"""
	def __init__(self, network_type, iterations, models):
		self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
		subject_set = self.VARS.get_subject_set(False)
		remove_activities = self.VARS.CONVERTION_ORIGINAL_INVERSE
		keep_activities = self.VARS.CONVERTION_ORIGINAL
		raw_signal_labels = get_raw_signal_labels(subject_set)
		output = 10
		window = "1.0"


		for i in range(0,len(models)):
			print "model: " 
			print i
			window_size = models[i][1]/6
			self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, output, remove_activities, None, keep_activities,  models[i][5])
			self.config = self.VARS.get_config(window_size, output, iterations, window_size, network_type, models[i][2][0], models[i][2][1], models[i][3][0], models[i][4])
		#	print 'Creating data set'
			
		#	self.cnn = CNN_MOD.CNN_MOD(config)
		#	self.cnn.set_data_set(self.data_set)

		#	self.cnn.load_model('models/' + models[i][0]+ '_' + str(models[i][1]) + '_' + str(models[i][2]) + '_' + str(models[i][3]) + '_' + str(models[i][4]) + '_' + models[i][5])

			
		#	print len(self.data_set.test._data)
			
		#	for j in range(0,len(self.data_set.test._data)):
		#		data = [self.data_set.test._data[j]]
		#		prediction = self.cnn.sess.run(self.cnn.y_conv, feed_dict={self.cnn.x: data ,self.cnn.keep_prob:1.0})
		#		for k in range(0,window_size):
		#			df_results.iloc[j*window_size+k][i*17:i*17+17]=prediction[0]
								
				
		#print df_results
		#np.savetxt('original_5_classifiers.csv', df_results, fmt='%s',delimiter=",")

		
        
#		correct_instances = 0.0;
#		for i in range(0,len(df_results.columns)):
#			if df_results.iloc[1][i] == int(raw_signal_labels[i]):
#				correct_instances += 1
#		print correct_instances
#		print correct_instances / len(df_results.columns)




#HENT MODELLENE
models = [["original",300,[20,40],[200],"SAME","0.5"],["original",450,[20,40],[200],"SAME","0.75"], ["original",600,[20,40],[200],"SAME","1.0"] ,["original",750,[20,40],[200],"SAME","1.5"]]

#,["sd",600,20,40,200,"SAME","1.0"],,["sd",1200,20,40,200,"VALID","2.0"]

cnn_h = CNN_TEST('original', 2000, models)


