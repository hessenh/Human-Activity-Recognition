import numpy as np


a = np.array([[1],[2],[3]])
print a


b = np.zeros([10])

c = len(b) / len(a)
print c

b =  np.tile(a, (c,1))

print b
print len(b)