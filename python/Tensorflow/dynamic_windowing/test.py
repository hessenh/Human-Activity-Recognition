
import numpy as np
import pandas as pd
filepath = 'original_5_classifiers.csv'
df = pd.read_csv(filepath, header=None, sep='\,',engine='python')
print df.iloc[0][0:5*17]
l = 5
data = np.zeros([l,5*17])
label = np.zeros([l,17])

for i in range(0,5):
	data[i] = df[i][0:5*17]
	label[i] = df[i][5*17]

print data
print label