import pandas as pd
import numpy as np
import math
predictions_sd = '../Tensorflow/predictions/prediction_sd_prob.csv'
actual_sd = '../Tensorflow/predictions/actual_sd_prob.csv'

observations = pd.read_csv(predictions_sd, header=None, sep='\,',engine='python').as_matrix()
actual_sd  = pd.read_csv(actual_sd, header=None, sep='\,',engine='python').as_matrix()
predictions_sd = observations.copy()


def generate_initial_prob(actual):
  length = len(actual)
  dynamic = sum(actual[:,0])
  static = sum(actual[:,1])
  return static/length, dynamic/length

def generate_transition_matrix(actual):
  s_d = 0
  s_s = 0
  d_s = 0
  d_d = 0
  for i in range(0,len(actual)-1):
    if actual[i][0] == 0 and actual[i+1][0] == 0:
      s_s +=1
    elif actual[i][0] == 0 and actual[i+1][0] == 1:
      s_d +=1
    elif actual[i][0] == 1 and actual[i+1][0] == 0:
      d_s +=1
    elif actual[i][0] == 1 and actual[i+1][0] == 1:
      d_d +=1
  n = (len(actual)-1)
  return s_d*1.0/n, s_s*1.0/n, d_s*1.0/n, d_d*1.0/n





''' Start Probability '''
static_prob, dynamic_prob =  generate_initial_prob(actual_sd)
start_probability = {'STATIC': math.log(static_prob), 'DYNAMIC': math.log(dynamic_prob)}

''' Transition Probability '''
s_d, s_s, d_s, d_d = generate_transition_matrix(actual_sd)
transition_probability = {
   'STATIC' : {'STATIC': math.log(s_s), 'DYNAMIC': math.log(s_d)},
   'DYNAMIC' : {'STATIC': math.log(d_s), 'DYNAMIC': math.log(d_d)}
   }

''' Change observations (emission) - Normalize and divide on class dist '''
for i in range(0,len(observations)):
  observations[i][0] = observations[i][0]/ start_probability['DYNAMIC']
  observations[i][1] = observations[i][1]/ start_probability['STATIC']
  s = sum(observations[i])
  observations[i][0] = math.log(observations[i][0]/s)
  observations[i][1] = math.log(observations[i][1]/s)

states = ['DYNAMIC','STATIC']

def viterbi(obs, states, start_p, trans_p):
  V = [{}]
  # Initialize base cases (t == 0)
  for y in range(len(states)):
    V[0][states[y]] = start_p[states[y]] + obs[0][y]
  # Run Viterbi for t > 0
  for t in range(1, len(obs)):
    V.append({})

    for y in range(len(states)):
      ''' 
      V[t-1][j] - Probability of state in t-1 (time). Example V[0][Healthy] = 0.3
      trans_p[j][y] - Transition Probability of going from state to state. Example V[Healthy][Healthy] = 0.7
      emit_p[y][obs[t]] - Emition Probability of seeing state given the observation. Example emit_p[Healthy][cold] = 0.4
      '''
      p = []
      for j in states:
        p.append(V[t-1][j] + trans_p[j][states[y]] + obs[t][y])
      prob = max(p)
      V[t][states[y]] = prob

  return V

import operator
def getMax(dict):
	return max(dict.iteritems(), key=operator.itemgetter(1))[0]

def example():
    return viterbi(observations,
                   states,
                   start_probability,
                   transition_probability)

prob = example();
score = 0

l = len(actual_sd)
for i in range(0,l):
  p =  getMax(prob[i])
  a = np.argmax(actual_sd[i])
  if p == 'STATIC' and a == 1:
    score += 1
  elif p == 'DYNAMIC' and a == 0:
    score +=1

print score*1.0 / l

score = 0
for i in range(0,l):
  p = np.argmax(predictions_sd[i])
  a = np.argmax(actual_sd[i])
  if p == a:
    score += 1
print score*1.0 / l