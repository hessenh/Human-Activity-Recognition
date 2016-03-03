

import numpy as np 

a = np.array([[1],[2],[1],[2],[3]])
data = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4],[7,7,7,7]])

r = [1]
for i in r:
	print data == (a != r)