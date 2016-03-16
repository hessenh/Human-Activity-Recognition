import pandas as pd
import numpy as np
import math

#Generate trans matrix on average duration of activities

def generateTransMatrix(activityLen,network_type):

	actual = './predictions/actual_'+network_type+'_prob.csv'
	actual_labels  = pd.read_csv(actual, header=None, sep='\,',engine='python').as_matrix()

	transition = np.zeros((activityLen,activityLen))
	
	actCount = np.zeros((2,activityLen))
	i=0

	while i<len(actual_labels)-1:
		act = np.argmax(actual_labels[i])
		sequenceLength=0
		while act == np.argmax(actual_labels[i]) and i<len(actual_labels)-1:
			sequenceLength=sequenceLength+1
			i=i+1
		actCount[0][act]=actCount[0][act]+1
		actCount[1][act]=actCount[1][act]+sequenceLength

	print actCount

	averageActLength = np.zeros(activityLen)

	for i in range(0,activityLen):
		averageActLength[i] = np.round(actCount[1][i]/actCount[0][i])
		for j in range(0,activityLen):
			if i == j:
				transition[i][j]=(averageActLength[i]-1) / averageActLength[i]
			else:
				transition[i][j] = 1/(averageActLength[i]*(activityLen-1))

	print averageActLength
	print transition

	return transition

