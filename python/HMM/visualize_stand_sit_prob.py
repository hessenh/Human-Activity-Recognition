# Plot predicted dynamic vs static vs original label
# 
# ==============================================================================


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
#ORIGINAL = './predictions/original.csv'

network_type = 'sd'

actual = 'predictions/actual_'+network_type+'_prob.csv'
PREDICTION= 'predictions/prediction_'+network_type+'_prob.csv'
VITERBI = 'predictions/viterbi_'+network_type+'.csv'
#df_original = pd.read_csv(ORIGINAL, header=None, sep='\,', engine='python'
df_actual = pd.read_csv(actual, header=None, sep='\,', engine='python')
df_prediction = pd.read_csv(PREDICTION, header=None, sep='\,', engine='python')
df_viterbi = pd.read_csv(VITERBI, header=None, sep='\,', engine='python')




#original = df_original.values
actual = df_actual.values
predictions = df_prediction.values
viterbi = df_viterbi.values


start = 0#1500
end = 1000

#original = original[start:end]

actual = actual[start:end]
predictions = predictions[start:end]
viterbi = viterbi[start:end]
print 


size = len(predictions)
scorePre= np.zeros(size)
scoreVit= np.zeros(size)
for i in range(0, end-start):
	pre = np.argmax(predictions[i])
	act = np.argmax(actual[i])
	if pre == act:
		scorePre[i] = 1
	if viterbi[i][0] == act:
		scoreVit[i] = 1

print 'CNN',sum(scorePre)*1.0 / (end-start)
print 'Viterbi',sum(scoreVit)*1.0 / (end-start)




actual_max = np.zeros(len(actual))
predictions_max = np.zeros(len(actual))

for i in range(0,len(actual)):
	actual_max[i] = np.argmax(actual[i])+1
	predictions_max[i] = np.argmax(predictions[i])+1

plt.figure(1)



plt.subplot(311)
axes = plt.gca()
axes.set_ylim([0.9,10.4])
plt.plot(actual_max)

plt.subplot(312)
axes = plt.gca()
axes.set_ylim([0.9,10.4])
plt.plot(predictions_max)

plt.subplot(313)
axes = plt.gca()
axes.set_ylim([0.9,10.4])
plt.plot(viterbi + 1)



plt.show()


