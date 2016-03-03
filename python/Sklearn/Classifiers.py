from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

import data_features
import VAR


def classify(classifier):
	VARIABLES = VAR.VARIABLES()
	# Get data set
	data = data_features.Data_Set()

	# Set up classifier
	if classifier == 'SVM':
		clf = svm.SVC()
	elif classifier == 'NearestCentroid':
		clf = NearestCentroid()
	elif classifier == 'RF':
		clf = RandomForestClassifier(n_estimators = 200 )
	elif classifier == 'SGD':
		clf = SGDClassifier(loss="hinge", penalty="l2")
	elif classifier == 'GNB':
		clf = GaussianNB()
	
	# Fit the training data to the labels and create the decision trees
	clf.fit(data.train_x,data.train_l)  


	# Take the same decision trees and run it on the test data
	keep_activities = VARIABLES.CONVERTION_ORIGINAL
	for activity in keep_activities:
	
		activity_data = data.test_x[data.test_original_l[0] == keep_activities[activity]]
		activity_label = data.test_l[data.test_original_l[0] == keep_activities[activity]]
		activity_score =  clf.score(activity_data, activity_label)
		print str(activity_score).replace(".",",")

	score = clf.score(data.test_x,data.test_l)
	print 'Total', str(score).replace(".",",")

classify('RF')