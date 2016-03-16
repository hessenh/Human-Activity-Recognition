import pandas as pd
import numpy as np
import math
import copy
from Viterbi import *

network_type = 'original'		
predictions = './predictions/prediction_'+network_type+'_prob.csv'
actual = './predictions/actual_'+network_type+'_prob.csv'

states = ['WALKING','RUNNING','STAIRS (UP)','STAIRS (DOWN)','STANDING','SITTING','LYING','BENDING','CYCLING (SITTING)','CYCLING (STANDING)']
numOfAct = len(states)

v = Viterbi(states)
v.load_observations(predictions)
actual_labels = v.load_actual_labels(actual)
v.generate_start_probability(numOfAct)


#transMatrix={'STANDING': {'STANDING': 82.0, 'BENDING': 3.0, 'WALKING': 7.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 2.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
#	'BENDING': {'STANDING': 23.0, 'BENDING': 69.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
#	'WALKING': {'STANDING': 14.0, 'BENDING': 1.0, 'WALKING': 78.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
#	'CYCLING (SITTING)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING':1.0, 'CYCLING (SITTING)': 89.0, 'SITTING': 3.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
#	'SITTING': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 91.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
#	'CYCLING (STANDING)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 91.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
#	'RUNNING': {'STANDING': 2.0, 'BENDING': 1.0, 'WALKING': 6.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 85.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
#	'STAIRS (UP)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 91.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
#	'STAIRS (DOWN)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 91.0, 'LYING': 1.0}, 
#	'LYING': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 91.0}}

transMatrix={'STANDING': {'STANDING': 91.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'BENDING': {'STANDING': 1.0, 'BENDING': 91.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'WALKING': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 91.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'CYCLING (SITTING)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING':1.0, 'CYCLING (SITTING)': 91.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'SITTING': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 91.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'CYCLING (STANDING)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 91.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'RUNNING': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 91.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'STAIRS (UP)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 91.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'STAIRS (DOWN)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 91.0, 'LYING': 1.0}, 
	'LYING': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 91.0}}


v.transition_probability = copy.deepcopy(transMatrix)
v.generate_transition_probability()
v.generate_observation_probability()
v.run()
v.generate_path()
score = v.get_accuracy()
viterbi = v.save_viterbi(network_type)
j=0
print len(actual_labels)
for i in range(j,len(actual_labels)-1):
	true_state = np.argmax(actual_labels[i])
	vit_state = viterbi[i]
	true_next_state = np.argmax(actual_labels[i+1])
	vit_next_state = viterbi[i+1]
	j=j+1
	if true_state==vit_state and true_next_state != vit_next_state:
		changeTransitionMatrix = True
		while(changeTransitionMatrix):
			print i
			transMatrix[states[true_state]][states[true_next_state]] =  transMatrix[states[true_state]][states[true_next_state]]  + 1.0
			transMatrix[states[vit_state]][states[vit_next_state]] =  transMatrix[states[vit_state]][states[vit_next_state]] - 1.0
			v.transition_probability = copy.deepcopy(transMatrix)
			v.generate_transition_probability()
			v.run()
			v.generate_path()
			new_score = v.get_accuracy()
			print score
			print new_score

			if new_score<=score:
				transMatrix[states[true_state]][states[true_next_state]] =  transMatrix[states[true_state]][states[true_next_state]]  - 1.0
				transMatrix[states[vit_state]][states[vit_next_state]] =  transMatrix[states[vit_state]][states[vit_next_state]] + 1.0
				changeTransitionMatrix = False
			else:
				score = new_score
			viterbi = v.save_viterbi(network_type)
		

print transMatrix