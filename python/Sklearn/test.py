import itertools

activities = ['walking'	
			,'Running'	
			,'shuffling'	
			,'stairs (ascending)'	
			,'stairs (descending)'	
			,'standing'	
			,'sitting'	
			,'lying'	
			,'Bending'	
			,'Picking'	
			,'Cycling (sitting)'	
			,'Cycling (stand)'	
			,'Vigorous Activities']

activities = ['walking'	
			,'Running'	
			,'shuffling'	
			,'stairs (ascending)'	
			,'stairs (descending)'	
			,'standing'	
			,'Vigorous Activities']

#activities = ["hei","hade","kanskj"]

subsets = []
for L in range(0, len(activities)+1):
  for subset in itertools.combinations(activities, L):
    subsets.append(subset)

subsets.pop(0)

print subsets[10]