import input_data_window_large
import CNN_MOD
import CNN_STATIC_VARIABLES
import numpy as np
import pandas as pd

def get_raw_signal_labels(subject_set,change_labels):
	print "Inni get raw signal labels"
	filepath = '../../../../Prosjektoppgave/Notebook/data/'+subject_set[1][0]+'/RAW_SIGNALS/'+subject_set[1][0]+'_GoPro_LAB_All.csv'
	df = pd.read_csv(filepath, header=None, sep='\,',engine='python')

	for i in range(1,len(subject_set[1])):
		filepath = '../../../../Prosjektoppgave/Notebook/data/'+subject_set[1][i]+'/RAW_SIGNALS/'+subject_set[1][i]+'_GoPro_LAB_All.csv'
		df_new= pd.read_csv(filepath,header=None, sep='\,',engine='python')
		df = pd.concat([df, df_new ], ignore_index=True)

	m = np.zeros(len(df))
	for i in range(len(df)):
		a = df.iloc[i]
		a = change_labels[a.values[0]]
		m[i]=a
	print len(m)
	return m   

class CNN_TEST(object):
	"""docstring for CNN_H"""
	def __init__(self, network_type, index, models):
		self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
		subject_set = self.VARS.get_subject_set()
		convertion = self.VARS.CONVERTION_STATIC_DYNAMIC
		raw_signal_labels = get_raw_signal_labels(subject_set,convertion)

 
		df_results = pd.DataFrame(index = np.arange(len(models)),columns=np.arange(len(raw_signal_labels)))

		for i in range(0,len(models)):

			window_size = models[i][1]/6
			config = self.VARS.get_config(models[i][1], 2, index, 100, models[i][0], models[i][2], models[i][3], models[i][4], models[i][5])
			print 'Creating data set'
			self.data_set = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None, models[i][6])

			self.cnn = CNN_MOD.CNN_MOD(config)
			self.cnn.set_data_set(self.data_set)

			self.cnn.load_model('models/' + models[i][0]+ '_' + str(models[i][1]) + '_' + str(models[i][2]) + '_' + str(models[i][3]) + '_' + str(models[i][4]) + '_' + models[i][5])

			print "hei"
			print len(self.data_set.test._data)
			print models[i][1]/6
			for j in range(0,len(self.data_set.test._data)):
				data = [self.data_set.test._data[j]]
				prediction = self.cnn.sess.run(self.cnn.y_conv, feed_dict={self.cnn.x: data ,self.cnn.keep_prob:1.0})
				prediction = np.argmax(prediction[0])+1
				index=j*window_size
				df_results.iloc[i][index:index+window_size] = prediction
		print df_results


		df_max = df_results.mode();
		correct_instances = 0.0;
		for i in range(0,len(df_max.columns)):
			if df_max[i][0] == int(raw_signal_labels[i]):
				correct_instances += 1
		print correct_instances
		print correct_instances / len(df_max.columns)
    		
        
#		correct_instances = 0.0;
#		for i in range(0,len(df_results.columns)):
#			if df_results[i][0] == int(raw_signal_labels[i]):
#				correct_instances += 1
#		print correct_instances
#		print correct_instances / len(df_results.columns)




#HENT MODELLENE
models = [["sd",900,20,40,200,"VALID","1.5"], ["sd",600,20,40,200,"VALID","1.0"]]

#["sd",600,20,40,200,"VALID","1.0"],["sd",300,20,40,200,"VALID","0.5"],["sd",1050,20,40,200,"VALID","1.75"],["sd",1200,20,40,200,"VALID","2.0"]

cnn_h = CNN_TEST('sd', 2000, models)


