import pandas as pd

overlap = 80
window_size = 100

''' Load viterbi output '''
viterbi_path = 'viterbi_sd.csv'
df_viterbi = pd.read_csv(viterbi_path, header=None, sep='\,',engine='python')


''' Load actual window labels '''
actual_path = '../../../Prosjektoppgave/Notebook/data/21A/DATA_WINDOW/1.0/ORIGINAL/GoPro_LAB_All_L.csv'
df_actual = pd.read_csv(actual_path, header=None, sep='\,',engine='python')

# Remove activities
remove = {9:9, 12:12, 17:17}
for key, value in remove.iteritems():
     df_actual =  df_actual[df_actual[0] != key]


# Convert activities
convertion = {1:1, 2:2, 3:11, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8, 11:8, 13:9, 14:10, 16:2}
for key, value in convertion.iteritems():
	df_actual[df_actual[0] == key] = value


''' Find shuffling '''
start = 0
stop = 0
start_stop = []
walking = False
for i in range(0, len(df_viterbi)):
	

	if df_viterbi.iloc[i].values[0] == 1.0 and not walking:
		walking = True
		start = i

	elif df_viterbi.iloc[i].values[0] != 1.0 and walking:
		walking = False
		stop = i
		#print start, stop
		if stop - start < 10:
			start_stop.append([start, stop])
		#start = i
print start_stop

for s in start_stop:
	start = s[0]
	stop = s[1]
	for i in range(start, stop):
		df_viterbi.iloc[i] = 11




score = 0
for i in range(0, len(df_viterbi)):
	if df_viterbi.iloc[i].values[0] == df_actual.iloc[i].values[0]:
		score+=1.0
	else:
		print df_viterbi.iloc[i].values[0], df_actual.iloc[i].values[0]

print score / len(df_viterbi)