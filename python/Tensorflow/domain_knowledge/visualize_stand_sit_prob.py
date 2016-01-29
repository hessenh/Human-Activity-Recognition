# Plot predicted dynamic vs static vs original label
# 
# ==============================================================================


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
ORIGINAL = '../predictions/original.csv'
actual_stand_sit = '../predictions/actual_stand_sit_prob.csv'
PREDICTION_STAND_SIT = '../predictions/prediction_stand_sit_prob.csv'
VITERBI_STAND_SIT = '../predictions/viterbi_stand_sit.csv'

df_original = pd.read_csv(ORIGINAL, header=None, sep='\,', engine='python')
df_actual_stand_sit = pd.read_csv(actual_stand_sit, header=None, sep='\,', engine='python')
df_prediction_stand_sit = pd.read_csv(PREDICTION_STAND_SIT, header=None, sep='\,', engine='python')
df_viterbi_stand_sit = pd.read_csv(VITERBI_STAND_SIT, header=None, sep='\,', engine='python')


RE_CONVERTION_SIT = {1:7,2:8}
RE_CONVERTION_OTHER = {1:10, 2:11, 3:13, 4:14, 5:15, 6:16, 7:17}
RE_CONVERTION_STAND = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:12}

original = df_original.values
actual_stand_sit = df_actual_stand_sit.values
predictions_stand_sit = df_prediction_stand_sit.values
viterbi_stand_sit = df_viterbi_stand_sit.values


start = 0
end = 750

original = original[start:end]

actual_stand_sit = actual_stand_sit[start:end]
predictions_stand_sit = predictions_stand_sit[start:end]
viterbi_stand_sit = viterbi_stand_sit[start:end]
print viterbi_stand_sit

size = len(predictions_stand_sit)
score_stand_sit = np.zeros(size)

for i in range(0, end-start):
	pre = np.argmax(predictions_stand_sit[i])
	act = np.argmax(actual_stand_sit[i])
	if pre == act:
		score_stand_sit[i] = 1

print 'Stand_Sit',sum(score_stand_sit)*1.0 / (end-start)







actual_stand_sit_max = np.zeros(len(actual_stand_sit))
predictions_stand_sit_max = np.zeros(len(actual_stand_sit))

for i in range(0,len(actual_stand_sit)):
	actual_stand_sit_max[i] = np.argmax(actual_stand_sit[i])+1
	predictions_stand_sit_max[i] = np.argmax(predictions_stand_sit[i])+1

plt.figure(1)



plt.subplot(511)
axes = plt.gca()
axes.set_ylim([0.9,3.1])
plt.plot(actual_stand_sit_max)
print actual_stand_sit_max

plt.subplot(512)
axes = plt.gca()
axes.set_ylim([0.9,3.1])
plt.plot(predictions_stand_sit_max)

plt.subplot(513)
axes = plt.gca()
axes.set_ylim([0.9,3.1])
plt.plot(viterbi_stand_sit + 1)

plt.subplot(514)
axes = plt.gca()
axes.set_ylim([0.9,17.1])
plt.plot(original)

plt.subplot(515)
axes = plt.gca()
axes.set_ylim([-0.2,1.2])
plt.plot(predictions_stand_sit[:,0], 'r--')
plt.plot(predictions_stand_sit[:,1], 'b--')
plt.plot(predictions_stand_sit[:,2], 'g--')

plt.show()


