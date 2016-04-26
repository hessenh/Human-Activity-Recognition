import pandas as pd
import numpy as np
import math


def baum_welch(number_activities,iterations,network_type):
	predictions = './predictions/PREDICTION_TRAINING.csv'
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
		

	trans1 = transition
	trans2 =countTransitions()
	print 'BW: ', trans1
	print 'overganger: ', trans2
	
	finalTrans = trans1 + trans2
	for i in range(0,10):
		a = finalTrans[i]/(np.sum(finalTrans[i])*1.0)
		finalTrans[i] = a.tolist()


	return finalTrans



def countTransitions():
	actual = './predictions/actual_'+'sd'+'_prob_train.csv'
	actual  = pd.read_csv(actual, header=None, sep='\,',engine='python').as_matrix()
	
	transition = np.zeros((10,10))
	transition = transition 
	
	for i in range(0,len(actual)-1):
		act = True
		prev = actual[i]
		prev = np.argmax(prev)
		if (prev == 10) or (prev ==11) or (prev == 12):
			act = False
		post = actual[i+1]
		post = np.argmax(post)
		if (post == 10) or (post ==11) or (post == 12):
			act = False
		if act == True:
			transition[prev][post] += 1
	
#	i=0
#	foundPrev = False
#	while i<len(actual):
#		
#		act = np.argmax(actual[i])
#
#		if foundPrev == False:
#			if (act != 10) and (act !=11) and (act != 12):
#				prev = act
#				foundPrev = True
#
#		else:
#			if (act != 10) and (act !=11) and (act != 12):
#				post = act
#				transition[prev][post] += 1
#				foundPrev = False
#		
#		i+=1


	for i in range(0,10):
		a = transition[i]/(np.sum(transition[i])*1.0)
		transition[i] = a.tolist()
	print transition
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
