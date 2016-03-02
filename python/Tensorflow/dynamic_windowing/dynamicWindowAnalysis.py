
import numpy as np
import pandas as pd


filepath = 'original_5_classifiers.csv'
df = pd.read_csv(filepath, header=None, sep='\,',engine='python')
#print df
length = 1#len(df)-200
start=5600
#sum prob
#print df.sum(axis=0) / len(df)


correctOptimal = 0.0
correctMajority = 0.0
correctWeigthed = 0.0
correctMajorityorWeigthed = 0.0
dfProb = df[[0,2,4,6,8]]
dfClass = df[[1,3,5,7,9]]


for i in range(start,start+length):
	majority = False
	weigthed = False
	optimal = False

	label_i = df.iloc[i][10]
	class_i = dfClass.iloc[i].values

	prob_i = dfProb.iloc[i].values
	print label_i
	print class_i
	print prob_i


	#Majority
	probOverTreshHold_i = prob_i>0.5
	print probOverTreshHold_i

	classOverTreshold_i = class_i[probOverTreshHold_i]

	print classOverTreshold_i
	classOverTreshold_i= np.int_(classOverTreshold_i)

	print np.argmax(np.bincount(classOverTreshold_i))

	















	


	mode = dfClass.iloc[i][0:5].mode()

	if mode.values[0] == label:
		correctMajority +=1.0
		majority = True

	#Weigthed
	if df.iloc[i][np.argmax(dfProb.iloc[i])+1] == label:
		correctWeigthed+=1;
		weigthed=True

	#Optimal		
	if np.sum(df.iloc[i][0:10] == label)>0:
		correctOptimal+=1.0
		optimal = True

	if majority or weigthed:
		correctMajorityorWeigthed+=1.0


print correctOptimal
print correctMajorityorWeigthed
print "optimal: ", correctOptimal/length
print "majority: ", correctMajority/length
print "weigthed: ", correctWeigthed/length
print "MajorOrWeigthed: ", correctMajorityorWeigthed/length