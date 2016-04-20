# Plot predicted dynamic vs static vs original label
# 
# ==============================================================================


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
#ORIGINAL = './predictions/original.csv'

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

start = 26500#1500
end = 27500#len(viterbi)

actual = actual[start:end]
predictions = predictions[start:end]
viterbi = viterbi[start:end]



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


