import pandas as pd
import numpy as np
import math
from baum_welch import baum_welch 
from baum_welch import countTransitions
from transMatrix import generateTransMatrix 
import pickle


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
		return self.actual_labels

	def generate_start_probability(self,numOfAct):
		self.start_probability = {}
		for i in range(len(self.states)):
			self.start_probability[self.states[i]] = 1/numOfAct #np.log(np.count_nonzero(self.actual_labels[:,i])*1.0 / self.actual_labels_length)
		print '- Generated Start Probability'


	def generate_transition_probability(self,transition):
		 # Create structure of matrix
		self.transition_probability = {}
		for i in range(0,len(self.states)):
			temp_dict = {}
			for j in range(0,len(self.states)):
				temp_dict[self.states[j]] = transition[i][j]/np.sum(transition[i])
			self.transition_probability[self.states[i]] = temp_dict
		print self.transition_probability
		#print self.transition_probability
		 #Update the transition matrix with values
		#for i in range(0, self.actual_labels_length-1):
		#	first =  self.states[np.argmax(self.actual_labels[i])]
		#	sec = self.states[np.argmax(self.actual_labels[i+1])]
		#	self.transition_probability[first][sec] += 1
		

		 #Divide the values in the matrix by the length of the observation
		for d in self.transition_probability:
			
			#labels = self.actual_labels[:,self.states.index(d)] 
		#	labelsCount = np.sum(labels)
			for key, value in self.transition_probability[d].items():
				self.transition_probability[d][key] = np.log(value) #/ labelsCount)
		print self.transition_probability
		print '- Generated Transition Probability'



	def generate_observation_probability(self):
		# For each observation

		for j in range(0, self.observations_length):#self.observations_length
			# Do it for evert state
			for i in range(0,len(self.states)):
				self.emission_probability[j][i] = self.observations[j][i]  / np.exp(self.start_probability[self.states[i]])
			s = np.sum(self.emission_probability[j])
			for i in range(0, len(self.states)):
				
				self.emission_probability[j][i] = self.emission_probability[j][i] / s
				

			for i in range(0, len(self.states)):
				
				self.emission_probability[j][i] = np.log(self.emission_probability[j][i])
				
		print '- Generated Emission Probability'
	
	def save_obj(self, obj, name ):
		with open('models/'+ name + '.pkl', 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

	def load_obj(self, name):
	    with open('models/' + name + '.pkl', 'rb') as f:
	        return pickle.load(f)

	def run(self):
		for y in range(len(self.states)):
			self.V[0][self.states[y]] = self.start_probability[self.states[y]] + self.emission_probability[0][y]
			self.path[self.states[y]] = [self.states[y]]
		for t in range(1, len(self.emission_probability)):
			self.V.append({})
			newPath = {}
			for y in range(0, len(self.states)):
				(prob, state) = max((self.V[t-1][y0] + self.transition_probability[y0][self.states[y]] + self.emission_probability[t][y], y0) for y0 in self.states)
				newPath[self.states[y]] = self.path[state] + [self.states[y]]
				self.V[t][self.states[y]] = prob

			self.path = newPath

	def generate_path(self):
		n = self.observations_length-1
		(prob, state) = max((self.V[n][y], y) for y in self.states)
		self.end_state = state
		return self.path[state]

	def get_accuracy(self,actLen):
		score_vit = 0
		score_cnn = 0
		activity_accuracy=np.zeros((3,13))
		
		for i in range(0, self.actual_labels_length):
			true_state = np.argmax(self.actual_labels[i])
			vit_state = self.states.index(self.path[self.end_state][i])
			cnn_state = np.argmax(self.observations[i])

			activity_accuracy[0][true_state] = activity_accuracy[0][true_state]+1

			if true_state == vit_state:
				activity_accuracy[1][true_state] = activity_accuracy[1][true_state] +1
				score_vit +=1.0
			if true_state == cnn_state:
				score_cnn +=1.0
				activity_accuracy[2][true_state] = activity_accuracy[2][true_state] +1



		print 'Viterbi score:',score_vit / self.actual_labels_length
		print 'CNN score:',score_cnn / self.actual_labels_length
		print 'Viterbi sensitivity: ',activity_accuracy[1]/activity_accuracy[0]
		print 'CNN sensitivity: ',activity_accuracy[2]/activity_accuracy[0]
		print 'Diff sensitivity:', (activity_accuracy[1]/activity_accuracy[0] - activity_accuracy[2]/activity_accuracy[0])*100
		return score_vit / self.actual_labels_length

	def save_viterbi(self,classification):
		path = self.generate_path()
		viterbi = []
		for i in range(0,len(path)):
			viterbi.append(self.states.index(path[i]) + 1)
			
		np.savetxt('./predictions/viterbi_' + classification + '.csv', viterbi, delimiter=",")
		return viterbi

def main():
	network_type = 'sd'		
	predictions = './predictions/prediction_'+network_type+'_prob_test_all.csv'
	actual = './predictions/actual_'+network_type+'_prob_test_all.csv'
	loading_models = True
	#states = ['STAND','SIT']
	#states = ['WALKING','RUNNING','SHUFFLING','STAIRS (UP)', 'STAIRS (DOWN)', 'STANDING', 'VIGOROUS', 'NON-VIGOROUS']
	#states = ['STAIRS UP', 'STAIRS DOWN']
	#states = ['STAIRS UP', 'STAIRS DOWN','WALK']
	#states = ['SHUF', 'STAND','NON-VIGOROUS']
	#states = ['S','D']
	states = ['WALKING','RUNNING','STAIRS (UP)','STAIRS (DOWN)','STANDING','SITTING','LYING','BENDING','CYCLING (SITTING)','CYCLING (STANDING)']


	numOfAct = len(states)
	v = Viterbi(states)
	v.load_observations(predictions)
	v.load_actual_labels(actual)

	if loading_models:
		print '- Loading models'
		v.start_probability =  v.load_obj('start_probability')
		v.transition_probability = v.load_obj('transition_probability')
		print v.transition_probability
		v.generate_observation_probability()
		print '- Running viterbi'
		v.run()
		print '- Generating path'
		v.generate_path()
		
		viterbi = v.save_viterbi(network_type)
		v.get_accuracy(numOfAct)

	else:
		v.generate_start_probability(numOfAct)
		v.save_obj(v.start_probability, 'start_probability')

		trans = baum_welch(len(states),5,network_type)
		
		#trans = countTransitions()
		print trans

		#v.transition_probability={'STANDING': {'STANDING': 82.0, 'BENDING': 3.0, 'WALKING': 7.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 2.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
		#'BENDING': {'STANDING': 23.0, 'BENDING': 69.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
		#'WALKING': {'STANDING': 14.0, 'BENDING': 1.0, 'WALKING': 78.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
		#'CYCLING (SITTING)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING':1.0, 'CYCLING (SITTING)': 89.0, 'SITTING': 3.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
		#'SITTING': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 91.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
		#'CYCLING (STANDING)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 91.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
		#'RUNNING': {'STANDING': 2.0, 'BENDING': 1.0, 'WALKING': 6.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 85.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
		#'STAIRS (UP)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 91.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
		#'STAIRS (DOWN)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 91.0, 'LYING': 1.0}, 
		#'LYING': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 91.0}}

		v.generate_transition_probability(trans)
		v.save_obj(v.transition_probability, 'transition_probability')

		v.generate_observation_probability()
		v.save_obj(v.emission_probability, 'emission_probability')

		v.run()
		v.generate_path()
		v.get_accuracy(numOfAct)
		viterbi = v.save_viterbi(network_type)
	
if __name__ == "__main__":
	main()