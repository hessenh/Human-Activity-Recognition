from sklearn.ensemble import RandomForestClassifier 
import data_features

# Get data set
data = data_features.Data_Set()

# Set up classifier
forest = RandomForestClassifier(n_estimators = 100 )

# Fit the training data to the labels and create the decision trees
forest = forest.fit(data.train_x,data.train_l)

# Take the same decision trees and run it on the test data
output = forest.predict(data.test_x)

# Calculate score with test data
score = forest.score(data.test_x,data.test_l)
print score

