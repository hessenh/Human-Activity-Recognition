from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import itertools
import data_features
import VAR
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

def classify_fit(classifier, window_size, keep_activities_dict = None, remove_activities_dict = None, keep_activities_original_dict = None, remove_activities_original_dict = None):
	VARIABLES = VAR.VARIABLES()
	# Get data set
	if keep_activities_dict == None:
		keep_activities_dict = VARIABLES.CONVERTION_GROUPED_ACTIVITIES
		remove_activities_dict = VARIABLES.CONVERTION_GROUPED_ACTIVITIES_INVERSE
		keep_activities_original_dict = VARIABLES.CONVERTION_ORIGINAL
		remove_activities_original_dict = VARIABLES.CONVERTION_ORIGINAL_INVERSE
	
	print "Loading data"
	data = data_features.Data_Set(keep_activities_dict, remove_activities_dict, keep_activities_original_dict, remove_activities_original_dict, False)

	# Set up classifier
	clf = select_classifier(classifier)

	# Fit the training data to the labels and create the decision trees
	print "Training model"
	clf.fit(data.train_x,data.train_l[0])

	model = classifier + '_' + str(window_size) + '_' + str(keep_activities_dict) + '.pkl'
	print "Saving model as", model
	joblib.dump(clf, 'models/'+ model, compress=1) 

	# Take the same decision trees and run it on the test data
	keep_activities = keep_activities_original_dict
	score_list = []
	# Remove duplicates
	activity_list = list(set(keep_activities.values()))
	for activity in activity_list:
		activity_data = data.test_x[data.test_original_l[0] == activity]
		activity_label = data.test_l[data.test_original_l[0] == activity]
		activity_score =  clf.score(activity_data, activity_label)
		score_list.append(activity_score)
		#print str(activity_score).replace(".",",")

	real = clf.score(data.test_x,data.test_l)
	overall = sum(score_list) / len(score_list)
	#print 'Real', str(real).replace(".",",")
	#print 'Overal', overall
	return overall, real


''' Testing classifier - returns sensitivity for each activity and total accuracy '''
def classify_test(classifier, window_size, test_type):
	print "Testing classifier", classifier
	VARIABLES = VAR.VARIABLES()
	# Get data set

	keep_activities_dict = VARIABLES.CONVERTION_GROUPED_ACTIVITIES
	remove_activities_dict = VARIABLES.CONVERTION_GROUPED_ACTIVITIES_INVERSE
	keep_activities_original_dict = VARIABLES.CONVERTION_ORIGINAL
	remove_activities_original_dict = VARIABLES.CONVERTION_ORIGINAL_INVERSE
	print "Loading data"
	data = data_features.Data_Set(keep_activities_dict, remove_activities_dict, keep_activities_original_dict, remove_activities_original_dict, True)

	# Set up classifier
	clf = select_classifier(classifier)
	model = classifier + '_' + str(window_size) + '_' + str(keep_activities_dict)
	print 'Loading model:', model
	clf = joblib.load('models/'+ model + '.pkl') 


	if test_type == "sensitivity":
		keep_activities = keep_activities_original_dict
		
		# Remove duplicates
		activity_list = list(set(keep_activities.values()))


		for activity in activity_list:
			activity_data = data.test_x[data.test_original_l[0] == activity]
			activity_label = data.test_l[data.test_original_l[0] == activity]
			activity_score =  clf.score(activity_data, activity_label)
			print str(activity_score).replace(".",",")

		score = clf.score(data.test_x,data.test_l)
		print 'Total', str(score).replace(".",",")

	elif test_type == "predict_labels":
		length = len(data.test_x)
		predictions = np.zeros([length,3])
		
		# Iterate over test samples
		for i in range(0,length):
			# Get the data point
			feature = data.test_x.iloc[i]
			label = data.test_l.iloc[i]
			prediction = clf.predict_proba(feature)
			index_of_max_prediction = np.argmax(prediction)
			predictions[i][0] = index_of_max_prediction + 1
			predictions[i][1] = prediction[0][index_of_max_prediction]
			predictions[i][2] = label

		df_predictions = pd.DataFrame(predictions)
		df_predictions.rename(columns={0: 'prediction', 1: 'probability' , 2: 'label'}, inplace=True)
		#df_predictions.columns = ["prediction","label"]
		df_predictions.to_csv("predictions/" + model + ".csv", index=None)



def select_classifier(classifier):
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
	elif classifier == 'AdaBoost':
		clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300)
	return clf


def generate_subset(keep_activities, remove_activities, size_of_subsets):
	# Generate the subsets
	subsets = []
	
	for subset in itertools.combinations(keep_activities, size_of_subsets):
		subsets.append(subset)

	return subsets


def generate_dictionaries(keep_activities, remove_activities, subset):
	# Create the convertion dictionary
	CONVERTION_GROUPED_ACTIVITIES = {}
	CONVERTION_ORIGINAL = {}
	for i in keep_activities:
		CONVERTION_GROUPED_ACTIVITIES[i] = len(subset) + 1
		CONVERTION_ORIGINAL[i] = i
	# Populate the dictionary
	group_nr = 1
	for i in subset:
		CONVERTION_GROUPED_ACTIVITIES[i] = group_nr
		group_nr +=1

	# Remove activity dictionary
	CONVERTION_GROUPED_ACTIVITIES_INVERSE = {}
	CONVERTION_ORIGINAL_INVERSE = {}
	for i in remove_activities:
		CONVERTION_GROUPED_ACTIVITIES_INVERSE[i] = i
		CONVERTION_ORIGINAL_INVERSE[i] = i

	return CONVERTION_GROUPED_ACTIVITIES, CONVERTION_GROUPED_ACTIVITIES_INVERSE, CONVERTION_ORIGINAL, CONVERTION_ORIGINAL_INVERSE

def subset_selector(keep_activities, remove_activities, size_of_subset_list):
	

	for size in size_of_subset_list:
		subsets = generate_subset(keep_activities, remove_activities, size)
		print "Size of subsets", size
		print 'Number of subsets', len(subsets)
		print subsets
		overall_result = []
		real_result = []

		# Iterate over all subsets
		for subset in subsets:
			CONVERTION_GROUPED_ACTIVITIES, CONVERTION_GROUPED_ACTIVITIES_INVERSE, CONVERTION_ORIGINAL, CONVERTION_ORIGINAL_INVERSE = generate_dictionaries(keep_activities, remove_activities, subset)
			overall, real = classify('RF', False, CONVERTION_GROUPED_ACTIVITIES, CONVERTION_GROUPED_ACTIVITIES_INVERSE, CONVERTION_ORIGINAL, CONVERTION_ORIGINAL_INVERSE)
			real_result.append(real)
			overall_result.append(overall)

		print "______________Finished_______________"
		real_index =  np.argmax(real_result)
		print "Real accuracy",real_result[real_index]
		print "Real subset", subsets[real_index]
		overall_index =  np.argmax(overall_result)
		print "Overall accuracy", overall_result[overall_index]
		print "Overall subset", subsets[overall_index]
		print "_____________________________________"


def plot_confusion_matrix(conf_arr, keep_activities,title='Confusion matrix'):
	#np.set_printoptions(precision=2)

	norm_conf = []
	for i in conf_arr:
	    a = 0
	    tmp_arr = []
	    a = sum(i, 0)
	    for j in i:
	        tmp_arr.append(float(j)/float(a))
	    norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf), cmap=plt.cm.summer, 
	                interpolation='nearest')

	width = len(conf_arr)
	height = len(conf_arr[0])

	for x in xrange(width):
	    for y in xrange(height):
	        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')

	cb = fig.colorbar(res)

	plt.title('Confusion Matrix')
	activities = ['Walking','Running','Shuffeling', 'Stairs (up)','Stairs (down)','Standing','Sitting','Lying','Transition','Bending','Picking','Cycling (sit)','Cycling (stand)', 'Vigorous','Non-vigorous']
	labels = []
	print keep_activities
	for key in keep_activities:
		print activities[keep_activities[key]-1]
		labels.append(activities[keep_activities[key]-1])

	#labels = ['Walking','Running','Shuffeling', 'Stairs (up)','Stairs (down)','Standing','Sitting','Lying','Transition','Bending','Picking','Cycling (sit)','Cycling (stand)', 'Vigorous','Non-vigorous']
	plt.xticks(range(width), labels,rotation='vertical')
	plt.yticks(range(height), labels)
	plt.show()




def main():
	subsets = False
	
	if subsets:
		
		keep_activities = [7,8,13]
		remove_activities = [1,2,3,4,5,6,9,10,11,12,14,15,16,17]
		size_of_subset_list = [1,2,3]
		subset_selector(keep_activities, remove_activities, size_of_subset_list)
	else:
		#classify_fit('RF', 1.0)
		classify_test('RF', 1.0, "predict_labels")
if __name__ == "__main__":
    main()