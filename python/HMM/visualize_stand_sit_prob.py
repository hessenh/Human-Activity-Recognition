# Plot predicted dynamic vs static vs original label
# 
# ==============================================================================


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import json
#ORIGINAL = './predictions/original.csv'

def get_score(result_matrix):
	activities = [0,1,2,3,4,5,6,7,8,9]
	'''TP / (FP - TP)
	Correctly classified walking / Classified as walking
	'''
	TP = np.zeros(len(activities))
	TN = np.zeros(len(activities))

	FP_TP = np.zeros(len(activities))
	TP_FN = np.zeros(len(activities))
	FP_TN = np.zeros(len(activities))
	
	actual = result_matrix[:,0]
	predicted = result_matrix[:,2]



	for activity in activities:
		''' FP - TP'''
		FP_TP[activity] = np.sum(predicted == activity) #len(df[df[0]==activity])

		''' TP - FN '''
		TP_FN[activity] = np.sum(actual == activity) #len(df_actual[df_actual[0]==activity])
		''' FP - TN '''
		FP_TN[activity] = np.sum(actual != activity)#len(df_actual[df_actual[0] != activity])

	for i in range(0, len(predicted)):
		if predicted[i] == actual[i]:
			TP[actual[i]] += 1.0
		
		for activity in activities:
			if actual[i] != activity and predicted[i]  != activity:
				TN[activity] += 1.0
				
	print FP_TP
	accuracy = sum(TP) / sum(TP_FN)
	specificity = TN / FP_TN
	precision = TP / FP_TP
	recall = TP / TP_FN
	return [accuracy, specificity, precision, recall]


def produce_statistics_json(result):
	print 'produce_statistics_json'
	score = get_score(result)

	ACTIVITY_NAMES_CONVERTION = {1:'WALKING',2:'RUNNING', 3:'STAIRS (UP)', 4:'STAIRS (DOWN)', 5:'STANDING', 6:'SITTING', 7:'LYING', 8:'BENDING', 9:'CYCLING (SITTING)', 10:'CYCLING (STANDING)'}

	specificity = {}
	precision = {}
	recall = {}
	for i in range(0, len(score[1])):
		specificity[ACTIVITY_NAMES_CONVERTION[i+1]] = score[1][i]
		precision[ACTIVITY_NAMES_CONVERTION[i+1]] = score[2][i]
		recall[ACTIVITY_NAMES_CONVERTION[i+1]] = score[3][i]

	statistics = {
		'ACCURACY' : score[0],
		'specificity': specificity,
		'PRECISION': precision,
		'RECALL': recall
	}
	path = 'TEST_STATISTICS.json'
	with open(path, "w") as outfile:
		json.dump(statistics, outfile)
	return statistics
		


network_type = 'sd'

actual = 'predictions/actual_'+network_type+'_prob_test_all.csv'
PREDICTION= 'predictions/prediction_'+network_type+'_prob_test_all.csv'
VITERBI = 'predictions/viterbi_'+network_type+'.csv'
#df_original = pd.read_csv(ORIGINAL, header=None, sep='\,', engine='python'
df_actual = pd.read_csv(actual, header=None, sep='\,', engine='python')
df_prediction = pd.read_csv(PREDICTION, header=None, sep='\,', engine='python')
df_viterbi = pd.read_csv(VITERBI, header=None, sep='\,', engine='python')




#original = df_original.values
actual = df_actual.values
predictions = df_prediction.values
viterbi = df_viterbi.values



predictions = np.argmax(predictions, axis=1)
actual = np.argmax(actual, axis=1)
viterbi = np.int_(viterbi.T[0])

keep_boolean = (actual!=10) & (actual!=11) & (actual!=12)



predictions = predictions[keep_boolean]
actual = actual[keep_boolean]
viterbi =viterbi[keep_boolean]






result = np.zeros((len(predictions), 3))
for i in range(0,len(result)):
	a = actual[i]
	c = predictions[i]
	v = viterbi[i]-1
	result[i] = [a,c,v]

produce_statistics_json(result)


#PLOTTING

#start = 0#1500
#end = len(predictions)





#actual = actual[start:end]
#predictions = predictions[start:end]
#viterbi = viterbi[start:end]

size = len(predictions)
scorePre= np.zeros(size)
scoreVit= np.zeros(size)
scoreActCNN= np.zeros(10)
scoreActVit = np.zeros(10)
realAct = np.zeros(10)
for i in range(0, size):
	realAct[actual[i]] = realAct[actual[i]] + 1
	if predictions[i] == actual[i]:
		scorePre[i] = 1
		scoreActCNN[actual[i]] = scoreActCNN[actual[i]] +1
	if viterbi[i] == actual[i]+1:
		scoreActVit[actual[i]] = scoreActVit[actual[i]] +1
		scoreVit[i] = 1

print 'CNN',sum(scorePre)*1.0 / size
print 'Viterbi',sum(scoreVit)*1.0 / size
print 'CNN: ',scoreActCNN/realAct
print 'Viterbi: ' ,scoreActVit/realAct
print 'diff: ',scoreActVit/realAct - scoreActCNN/realAct





plt.figure(1)

plt.subplot(311)
axes = plt.gca()
axes.set_ylim([0.5,11.5])
plt.plot(actual+1)

plt.subplot(312)
axes = plt.gca()
axes.set_ylim([0.5,10.5])
plt.plot(predictions+1)

plt.subplot(313)
axes = plt.gca()
axes.set_ylim([0.5,10.5])
plt.plot(viterbi)



plt.show()