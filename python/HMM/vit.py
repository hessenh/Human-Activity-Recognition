

states = ('Healthy', 'Fever')
 
observations = ('normal', 'cold', 'dizzy')
 
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
 
transition_probability = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
   }
 
emission_probability = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
   }

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})

        for y in states:
        	p = []
        	for j in states:
        		''' 
        		V[t-1][j] - Probability of state in t-1 (time). Example V[0][Healthy] = 0.3
        		trans_p[j][y] - Transition Probability of going from state to state. Example V[Healthy][Healthy] = 0.7
        		emit_p[y][obs[t]] - Emition Probability of seeing state given the observation. Example emit_p[Healthy][cold] = 0.4
        		'''
        		p.append(V[t-1][j] * trans_p[j][y] * emit_p[y][obs[t]])
        	prob = max(p)

        	V[t][y] = prob
    
    print V
    return V

import operator
def getMax(dict):
	return max(dict.iteritems(), key=operator.itemgetter(1))[0]

def example():
    return viterbi(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)

for i in example():
	print getMax(i)