import pandas as pd
import numpy as np
import math



class Viterbi(object):
	"""docstring for Viterbi"""
	def __init__(self, states):
		print '- Initializing Viterbi'
		self.states = states
		self.V = [{}]
  		self.path={}

	def load_observations(self, path):
		self.observations = pd.read_csv(path, header=None, sep='\,',engine='python').as_matrix()
		self.emission_probability = np.copy(self.observations)
		print '- Loaded observations'
		self.observations_length = len(self.observations)

	def load_actual_labels(self, path):
		self.actual_labels  = pd.read_csv(path, header=None, sep='\,',engine='python').as_matrix()
		print '- Loaded actual labels'
		self.actual_labels_length =  len(self.actual_labels)

	def generate_start_probability(self,numOfAct):
		self.start_probability = {}
		for i in range(len(self.states)):
			self.start_probability[self.states[i]] = 1/numOfAct #np.log(np.count_nonzero(self.actual_labels[:,i])*1.0 / self.actual_labels_length)
		print '- Generated Start Probability'


	def generate_transition_probability(self):
		# Create structure of matrix
		self.transition_probability = {}
		for state0 in self.states:
			temp_dict = {}
			for state1 in self.states:
				temp_dict[state1] = 0.0
			self.transition_probability[state0] = temp_dict
		
		# Update the transition matrix with values
		for i in range(0, self.actual_labels_length-1):
			first =  self.states[np.argmax(self.actual_labels[i])]
			sec = self.states[np.argmax(self.actual_labels[i+1])]
			self.transition_probability[first][sec] += 1
		# Divide the values in the matrix by the length of the observation
		for d in self.transition_probability:
			
			labels = self.actual_labels[:,self.states.index(d)] 
			labelsCount = np.sum(labels)
			for key, value in self.transition_probability[d].items():
				self.transition_probability[d][key] = np.log(value*1.0 / labelsCount)
    	print '- Generated Transition Probability'


	def generate_observation_probability(self):
		# For each observation

		for j in range(0, self.observations_length):#self.observations_length
			# Do it for evert state
			for i in range(0,len(self.states)):
				self.emission_probability[j][i] = self.observations[j][i]  / np.exp(self.start_probability[states[i]])
			s = np.sum(self.emission_probability[j])
			for i in range(0, len(self.states)):
				
				self.emission_probability[j][i] = self.emission_probability[j][i] / s
				

			for i in range(0, len(self.states)):
				
				self.emission_probability[j][i] = np.log(self.emission_probability[j][i])
				
		print '- Generated Emission Probability'
		


	def run(self):
		for y in range(len(self.states)):
			self.V[0][self.states[y]] = self.start_probability[self.states[y]] + self.emission_probability[0][y]
			self.path[self.states[y]] = [self.states[y]]
		for t in range(1, len(self.emission_probability)):
			self.V.append({})
			newPath = {}
			for y in range(0, len(self.states)):
				(prob, state) = max((self.V[t-1][y0] + self.transition_probability[y0][self.states[y]] + self.emission_probability[t][y], y0) for y0 in self.states)
				newPath[self.states[y]] = self.path[state] + [states[y]]
				self.V[t][self.states[y]] = prob
			self.path = newPath

	def generate_path(self):
		n = self.observations_length-1
		(prob, state) = max((self.V[n][y], y) for y in self.states)
		self.end_state = state
		return self.path[state]

	def get_accuracy(self):
		score_vit = 0
		score_cnn = 0

		for i in range(0, self.actual_labels_length):
			true_state = np.argmax(self.actual_labels[i])
			vit_state = self.states.index(self.path[self.end_state][i])
			cnn_state = np.argmax(self.observations[i])

			if true_state == vit_state:
				score_vit +=1.0
			if true_state == cnn_state:
				score_cnn +=1.0

		print 'Viterbi score:',score_vit / self.actual_labels_length
		print 'CNN score:',score_cnn / self.actual_labels_length

	def save_viterbi(self,classification):
		path = v.generate_path()
		viterbi = []
		for i in range(0,len(path)):
			viterbi.append(self.states.index(path[i]))
			
		np.savetxt('./predictions/viterbi_' + classification + '.csv', viterbi, delimiter=",")


network_type = 'cycling-sitting'		
predictions = './predictions/prediction_'+network_type+'_prob.csv'
actual = './predictions/actual_'+network_type+'_prob.csv'
states = ['STAND','SIT']
#states = ['WALKING','RUNNING','SHUFFLING','STAIRS (UP)', 'STAIRS (DOWN)', 'STANDING', 'VIGOROUS', 'NON-VIGOROUS']
#states = ['STAIRS UP', 'STAIRS DOWN']
#states = ['STAIRS UP', 'STAIRS DOWN','WALK']
states = ['SHUF', 'STAND','NON-VIGOROUS']
#states = ['S','D']
#states = ['WALKING','RUNNING','SHUFFLING','STAIRS (UP)','STAIRS (DOWN)','STANDING','SITTING','LYING','TRANSITION','BENDING','PICKING','UNDEFINED','CYCLING (SITTING)','CYCLING (STANDING)','HEEL-DROP','VIGOROUS','NON-VIGOROUS']
numOfAct = len(states)
v = Viterbi(states)
v.load_observations(predictions)
v.load_actual_labels(actual)
v.generate_start_probability(numOfAct)
v.generate_transition_probability()
v.generate_observation_probability()
v.run()
v.generate_path()
v.get_accuracy()
v.save_viterbi(network_type)
