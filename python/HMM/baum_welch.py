import pandas as pd
import numpy as np
import math


def baum_welch(number_activities,iterations,network_type):
	predictions = './predictions/prediction_'+network_type+'_prob_train.csv'
	predictions  = pd.read_csv(predictions, header=None, sep='\,',engine='python').as_matrix()
	
	transition = np.zeros((number_activities,number_activities))
	transition = transition + 1.0/number_activities
	
	predictions = np.log(predictions)
	

	for i in range(0,iterations):
		oldTransition = transition
		transition = np.log(transition)
		print "Start forward backward"
		forward_prob = forward(predictions,transition,number_activities)
		print "Done with forward"
		backward_prob = backward(predictions,transition,number_activities)
		print "Done with backward"

		for act1 in range(0,number_activities):
			for act2 in range(0,number_activities):
				a = 0
				b = 0 
				for i in range(0,len(predictions)-2): #loop through seq
					
					a = a + np.exp(forward_prob[i][act1]+backward_prob[i][act1]-backward_prob[0][act1])
					b = b + np.exp(forward_prob[i][act1]+backward_prob[i+1][act2]+transition[act1][act2]+predictions[i+1][act2]-backward_prob[0][act1])
				

					#print np.exp(forward_prob[i][act1]+backward_prob[i][act1])
				a=backward_prob[0][act1]+np.log(a)
				
				b=backward_prob[0][act1]+np.log(b)

				transition[act1][act2]=np.exp(b-a)

			
		for i in range(0,number_activities):
			transition[i]=transition[i]/sum(transition[i])
		
		#diffTransition = np.subtract(transition,oldTransition)
		#print np.abs(np.sum(diffTransition))

		# Umulige aktiviteter
	#Walking
	#transition[0][6] = 0.000000000000001
	#transition[0][8] = 0.000000000000001
	#transition[0][9] = 0.000000000000001
	
	#Running
	#transition[1][5] = 0.000000000000001
	#transition[1][6] = 0.000000000000001
	#transition[1][7] = 0.000000000000001
	#transition[1][8] = 0.000000000000001
	#transition[1][9] = 0.000000000000001

	#Stairs (up)
	#transition[2][5] = 0.000000000000001
	transition[2][6] = 0.0000000001
	transition[2][7] = 0.0000000001
	transition[2][8] = 0.0000000001
	#transition[2][9] = 0.000000000000001

	#Stairs (down)
	#transition[3][5] = 0.000000000000001
	#transition[3][6] = 0.000000000000001
	#transition[3][7] = 0.000000000000001
	#transition[3][8] = 0.000000000000001
	transition[3][9] = 0.0000000001

	#Sitting
	#transition[5][1] = 0.000000000000001
	transition[5][2] = 0.0000000001
	#transition[5][3] = 0.000000000000001

	#Lying
	#transition[6][0] = 0.000000000000001
	#transition[6][1] = 0.000000000000001
	#transition[6][2] = 0.000000000000001
	#transition[6][3] = 0.000000000000001
	transition[6][7] = 0.0000000001
	#transition[6][8] = 0.000000000000001
	#transition[6][9] = 0.000000000000001

	#Bending
	#transition[7][1] = 0.000000000000001
	transition[7][2] = 0.0000000001
	#transition[7][3] = 0.000000000000001
	#transition[7][6] = 0.000000000000001
	#transition[7][8] = 0.000000000000001
	transition[7][9] = 0.0000000001


	#Cycl (sit)
	#transition[8][0] = 0.000000000000001
	#transition[8][1] = 0.000000000000001
	#transition[8][2] = 0.000000000000001
	#transition[8][3] = 0.000000000000001
	#transition[8][6] = 0.000000000000001
	#transition[8][7] = 0.000000000000001

	#Cycl (stand)
	#transition[9][0] = 0.000000000000001
	#transition[9][1] = 0.000000000000001
	transition[9][2] = 0.0000000001
	#transition[9][3] = 0.000000000000001
	#transition[9][6] = 0.000000000000001
	#transition[9][7] = 0.000000000000001


	return transition




def forward(predictions_log,transition_log, number_activities):
	forward_prob = np.zeros((len(predictions_log),number_activities))
	
	forward_prob[0] = np.log(1.0/number_activities)
	for t in range(1,len(forward_prob)):
		for act in range(0,number_activities):
			maxProb = 0
			maxProbIndex = 0
			prob = 0
			for prev_act in range(0,number_activities):		
				if forward_prob[t-1][prev_act]+transition_log[prev_act][act]+predictions_log[t][act]<maxProb:
					maxProb = forward_prob[t-1][prev_act]+transition_log[prev_act][act]+predictions_log[t][act]
					maxProbIndex = prev_act
		
			for prev_act in range(0,number_activities):
				prob = prob + np.exp(forward_prob[t-1][prev_act]+transition_log[prev_act][act]+predictions_log[t][act]-maxProb)

			prob_t = maxProb + np.log(prob)

			forward_prob[t][act] = prob_t
				
	return forward_prob

def backward(predictions_log,transition_log, number_activities):
	backward_prob = np.zeros((len(predictions_log),number_activities))
	backward_prob[len(backward_prob)-1] = np.log(1.0/number_activities)
	for t in range(len(backward_prob)-2,-1,-1):
		for act in range(0,number_activities):
			maxProb = 0
			maxProbIndex = 0
			prob = 0
			for next_act in range(0,number_activities):
				if backward_prob[t+1][next_act]+transition_log[act][next_act]+predictions_log[t][act]<maxProb:
					maxProb = backward_prob[t+1][next_act]+transition_log[act][next_act]+predictions_log[t][act]
					maxProbIndex = next_act
			for next_act in range(0,number_activities):
				
				prob = prob + np.exp(backward_prob[t+1][next_act]+transition_log[act][next_act]+predictions_log[t][act]-maxProb)

			prob_t = maxProb + np.log(prob)	

			backward_prob[t][act] = prob_t
			
	return backward_prob
