import numpy as np

train = np.array([[1,2],[2,2],[3,2]])
train_lab = np.array([[0,1],[0,1],[1,0]])

test = np.array([[4, 5],[5,6],[6,7]])

p = np.array([2,1])


print train, 'train'
print train_lab,'train_lab'
print test, 'test'


temp_lab = np.zeros(len(p))
temp_lab[p[1]]= 1
temp_data = test[p[0]]

test = np.delete(test,2,axis=0)
train = np.insert(train,len(train), temp_data, axis=0)
train_lab = np.insert(train_lab, len(train_lab), temp_lab, axis = 0)



print train, 'train'
print train_lab,'train_lab'
print test, 'test'
