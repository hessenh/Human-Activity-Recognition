# Plot predicted dynamic vs static vs original label
# 
# ==============================================================================


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
ORIGINAL = '../predictions/original.csv'
actual_SD = '../predictions/actual_sd_prob.csv'
PREDICTION_SD = '../predictions/prediction_sd_prob.csv'
PREDICTION_STATIC = '../predictions/prediction_static_prob.csv'
PREDICTION_DYNAMIC = '../predictions/prediction_dynamic_prob.csv'

df_original = pd.read_csv(ORIGINAL, header=None, sep='\,', engine='python')
df_actual_sd_sd = pd.read_csv(actual_SD, header=None, sep='\,', engine='python')
df_prediction_sd = pd.read_csv(PREDICTION_SD, header=None, sep='\,', engine='python')
df_prediction_static = pd.read_csv(PREDICTION_STATIC, header=None, sep='\,', engine='python')
df_prediction_dynamic = pd.read_csv(PREDICTION_DYNAMIC, header=None, sep='\,', engine='python')

RE_CONVERTION_STATIC = {1:6, 2:7, 3:8, 4:16, 5:17}
RE_CONVERTION_DYNAMIC = {1:1, 2:2, 3:3, 4:4, 5:5, 6:9, 7:10, 8:11, 9:12, 10:13, 11:14, 12:15}

original = df_original.values
actual_sd = df_actual_sd_sd.values
predictions_sd = df_prediction_sd.values
predictions_static = df_prediction_static.values
predictions_dynamic = df_prediction_dynamic.values


start = 0
end = 2399

original = original[start:end]
actual_sd = actual_sd[start:end]
predictions_sd = predictions_sd[start:end]
predictions_static = predictions_static[start:end]
predictions_dynamic = predictions_dynamic[start:end]

size = len(predictions_sd)
score_sd = np.zeros(size)

for i in range(0, end-start):
	pre = np.argmax(predictions_sd[i])
	act = np.argmax(actual_sd[i])
	if pre == act:
		score_sd[i] = 1

print 'SD',sum(score_sd)*1.0 / (end-start)

predictions_static_new = np.zeros(size)
predictions_static_prob = np.zeros(size)
predictions_dynamic_new = np.zeros(size)
predictions_dynamic_prob = np.zeros(size)
predictions_final = np.zeros(size)
error = np.zeros(size)

for i in range(0,size):
	actual_sd[i] = actual_sd[i][0]
	predictions_sd[i] = predictions_sd[i][0]
	# STATIC VS DYNAMIC
	# If the probability for an activity is close to 50/50 between to "sure" activities, let it be the average between them
	#if i>0 and i<size-1 and predictions_sd[i][0] > 0.1 and predictions_sd[i][0] < 0.7:
	#	if predictions_sd[i-1][0] > 0.8 and predictions_sd[i+1][0] > 0.8:
	#		predictions_sd[i] = (predictions_sd[i-1] + predictions_sd[i+1])/2

	predictions_static_index = np.argmax(predictions_static[i]) 
	predictions_static_prob[i] = predictions_static[i][predictions_static_index]
	predictions_static_new[i] = RE_CONVERTION_STATIC.get(predictions_static_index+1)

	predictions_dynamic_index = np.argmax(predictions_dynamic[i]) 
	predictions_dynamic_prob[i] = predictions_dynamic[i][predictions_dynamic_index]
	predictions_dynamic_new[i] = RE_CONVERTION_DYNAMIC.get(predictions_dynamic_index+1)
	if predictions_sd[i][0] > 0.5:
		predictions_final[i] = predictions_dynamic_new[i]
	else:
		predictions_final[i] = predictions_static_new[i]

score = 0
error_values = np.zeros(17)
for i in range(0, end-start):
	if original[i] == predictions_final[i]:
		score +=1
	else:
		error[i] = predictions_final[i]
		error_values[error[i]-1] +=1

for i in range(0,len(error_values)):
	print 'Act: ',i+1, 'Error:',error_values[i],'Count', list(original).count(i+1)



print 'Total', score*1.0 / (end-start)

predictions_sd_argmax = []
for i in range(0,size):
	predictions_sd_argmax.append(round(predictions_sd[i][0]))

np.savetxt('prediction.csv', predictions_final, delimiter=",")

plt.figure(1)

plt.subplot(711)
axes = plt.gca()
axes.set_ylim([-0.1, 1.1])
plt.plot(actual_sd)

plt.subplot(712)
axes = plt.gca()
axes.set_ylim([-0.1, 1.1])
plt.plot(predictions_sd_argmax)
plt.plot(predictions_sd)
#plt.plot(new_prediction)

plt.subplot(713)
axes = plt.gca()
axes.set_ylim([0, 17])
plt.plot(original)

plt.subplot(714)
axes = plt.gca()
axes.set_ylim([0, 17])
plt.plot(predictions_static_new, 'r--')
plt.plot(predictions_dynamic_new,  'b--')
plt.plot(predictions_final)

plt.subplot(715)
plt.plot(error)

plt.subplot(716)
axes = plt.gca()
axes.set_ylim([-0.1, 1.1])
plt.plot(predictions_static_prob)


plt.subplot(717)
axes = plt.gca()
axes.set_ylim([-0.1, 1.1])
plt.plot(predictions_dynamic_prob)



plt.show()


