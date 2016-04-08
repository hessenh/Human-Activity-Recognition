import numpy as np
import pandas as pd


activity_list = [0,1,2]


a = [[3987, [[  1.78081281e-02,   6.39465816e-06,   2.31705792e-03,
          6.54802352e-05,   9.79682505e-01,   3.44474283e-05,
          2.57732381e-06,   8.13340885e-05,   7.31739135e-07,
          1.31768581e-06]]], [8831, [[  1.67283622e-04,   6.22711980e-07,   7.68159452e-07,
          2.21226006e-08,   4.23783931e-04,   9.99343097e-01,
          2.13519215e-05,   4.23161509e-06,   3.87103028e-05,
          1.00941222e-07]]], [1902, [[  8.33926558e-01,   1.61741351e-04,   5.99241490e-03,
          2.80575850e-03,   1.54566675e-01,   5.00633672e-04,
          2.88874198e-05,   1.83554052e-03,   1.80877789e-04,
          9.55312998e-07]]], [2719, [[  9.13314879e-01,   6.71879434e-06,   2.07876321e-02,
          1.60926924e-04,   6.56995922e-02,   5.71994406e-06,
          3.64622252e-07,   8.60733871e-06,   1.54828849e-05,
          1.96155511e-07]]], [404, [[  2.98582449e-06,   1.83277482e-08,   6.92377536e-12,
          2.39163515e-11,   9.89483283e-07,   7.25656764e-06,
          9.99988794e-01,   4.09687256e-10,   1.04112179e-11,
          2.31293652e-12]]]]

predictions = np.zeros([len(a),3])
#print predictions

for i in range(0,len(a)):
	index = a[i][0]
	#print index
	prediction =  np.argmax(a[i][1])
	#print prediction
	activity = a[i][1][0][prediction]
	#print activity
	predictions[i] = [index, prediction, activity]



predictions =  predictions[predictions[:,2].argsort()]
predictions = predictions[predictions[:,2] > 0.9]
predictions = predictions[predictions[:,2] < 0.95]
print predictions
