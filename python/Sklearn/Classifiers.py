from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

import data_features


def classify(classifier):
	# Get data set
	data = data_features.Data_Set()
	# Set up classifier
	if classifier == 'SVM':
		clf = svm.SVC()
	elif classifier == 'NearestCentroid':
		clf = NearestCentroid()
	elif classifier == 'RF':
		clf = RandomForestClassifier(n_estimators = 100 )
	elif classifier == 'SGD':
		clf = SGDClassifier(loss="hinge", penalty="l2")
	elif classifier == 'GNB':
		clf = GaussianNB()
	
	#for i in range(0,len(data.train_x)):
	#	for j in range(0,len(data.train_x[i])):
	#		if data.train_x[i][j] < 0.0:
	#			print data.train_x[i][j]

	# Fit the training data to the labels and create the decision trees
	clf.fit(data.train_x,data.train_l)  

	# Take the same decision trees and run it on the test data
	#output = clf.predict(data.test_x)

	# Calculate score with test data
	score = clf.score(data.test_x,data.test_l)
	print score

classify('RF')