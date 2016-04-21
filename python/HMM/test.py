import numpy as np

transition_probability={'STANDING': {'STANDING': 82.0, 'BENDING': 3.0, 'WALKING': 7.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 2.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'BENDING': {'STANDING': 23.0, 'BENDING': 69.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'WALKING': {'STANDING': 14.0, 'BENDING': 1.0, 'WALKING': 78.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'CYCLING (SITTING)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING':1.0, 'CYCLING (SITTING)': 89.0, 'SITTING': 3.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'SITTING': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 91.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'CYCLING (STANDING)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 91.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'RUNNING': {'STANDING': 2.0, 'BENDING': 1.0, 'WALKING': 6.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 85.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'STAIRS (UP)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 91.0, 'STAIRS (DOWN)': 1.0, 'LYING': 1.0}, 
	'STAIRS (DOWN)': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 91.0, 'LYING': 1.0}, 
	'LYING': {'STANDING': 1.0, 'BENDING': 1.0, 'WALKING': 1.0, 'CYCLING (SITTING)': 1.0, 'SITTING': 1.0, 'CYCLING (STANDING)': 1.0, 'RUNNING': 1.0, 'STAIRS (UP)': 1.0, 'STAIRS (DOWN)': 1.0, 'LYING': 91.0}}


print transition_probability['STANDING']['BENDING']


states = ['WALKING','RUNNING','STAIRS (UP)','STAIRS (DOWN)','STANDING','SITTING','LYING','BENDING','CYCLING (SITTING)','CYCLING (STANDING)']


transition_probability[states[1]][states[0]] =  transition_probability[states[1]][states[0]]  + 1


print transition_probability[states[1]][states[0]]



transition1 = [
[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
[0.01,0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
[0.01,0.01,0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
[0.01,0.01,0.01,0.91,0.01,0.01,0.01,0.01,0.01,0.01],
[0.01,0.01,0.01,0.01,0.91,0.01,0.01,0.01,0.01,0.01],
[0.01,0.01,0.01,0.01,0.01,0.91,0.01,0.01,0.01,0.01],
[0.01,0.01,0.01,0.01,0.01,0.01,0.91,0.01,0.01,0.01],
[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.91,0.01,0.01],
[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.91,0.01],
[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.91],]

transition2 = [
[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
[0.01,0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
[0.01,0.01,0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
[0.01,0.01,0.01,0.91,0.01,0.01,0.01,0.01,0.01,0.01],
[0.01,0.01,0.01,0.01,0.91,0.01,0.01,0.01,0.01,0.01],
[0.01,0.01,0.01,0.01,0.01,0.91,0.01,0.01,0.01,0.01],
[0.01,0.01,0.01,0.01,0.01,0.01,0.91,0.01,0.01,0.01],
[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.91,0.01,0.01],
[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.91,0.01],
[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.91],]

trans = [[1,2,3],[2,3,4],[1,1,1]]
print np.sum(trans[0])
a = trans[0]/(np.sum(trans[0])*1.0)
trans[0] = a.tolist()
print trans
#transition = np.zeros((10,10))
#transition = transition + 1.0/10
#print np.subtract(transition1,transition2) 



hi = np.zeros((5,10))

hi[0]=1

