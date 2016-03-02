''' Used to understand the CNN structure'''

import tensorflow as tf
# sensor data input format = [x,x,x,x,y,y,y,y,z,z,z,z,...x,x,x,x,y,y,y,y,z,z,z,z,]
window = 24
axes = 6
size_input = window*axes
#x = tf.Variable([1,2,3,4,5,6,7,8,9])
x = tf.placeholder("float", shape=[None, size_input])
y_ = tf.placeholder("float", shape=[None, 17])
print x.get_shape()
nn_size = 1024
features_l_1 = 20
features_l_2 = 40
''' Forste lag '''
# w
# Patch size, input channels, output channels
w_1 = tf.truncated_normal([6,6,1,features_l_1], stddev=0.1)
w_1 =  tf.Variable(w_1)
print w_1.get_shape(), 'Conv filters 1'
# b
b_1 = tf.constant(0.1, shape=[features_l_1])
b_1 = tf.Variable(b_1)

# Data points
x_image = tf.reshape(x, [-1, 6, window, 1])
print x_image.get_shape(),'reshape'
# First conv
h_1 = tf.nn.conv2d(x_image, w_1, strides=[1,1,1,1], padding='VALID')
h_1 = tf.nn.relu(h_1 + b_1)
print h_1.get_shape(), "Features 1"

w_2 = tf.truncated_normal([1,6,features_l_1,features_l_2], stddev=0.1)
w_2 = tf.Variable(w_2)
print w_2.get_shape(), 'Conv filter 2'
b_2 = tf.constant(0.1, shape=[features_l_2])
b_2 = tf.Variable(b_2)
h_2 = tf.nn.conv2d(h_1, w_2, strides=[1, 1, 1, 1], padding='VALID')
print h_2.get_shape(), 'Features 2'


h_p2_flat = tf.reshape(h_2, [-1, 1*14*features_l_2])
print h_p2_flat.get_shape(), 'Output conv'

# ''' NN '''
w_fc1 = tf.truncated_normal([1*(window-5-5)*features_l_2,nn_size])
w_fc1 = tf.Variable(w_fc1)
print w_fc1.get_shape(), 'Neural network input'
b_fc1 = tf.constant(0.1, shape=[nn_size])
b_fc1 = tf.Variable(b_fc1)

h_fc1 = tf.matmul(h_p2_flat, w_fc1)
h_fc1 = tf.nn.relu(h_fc1 + b_fc1)

w_fc2 = tf.truncated_normal([nn_size,17])
w_fc2 = tf.Variable(w_fc2)
print w_fc2.get_shape()
b_fc2 = tf.constant(0.1, shape=[17])
b_fc2 = tf.Variable(b_fc2)
print b_fc2.get_shape()

y_conv = tf.matmul(h_fc1, w_fc2)
y_conv = tf.nn.softmax(y_conv + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
data = []
for i in range(1,size_input+1):
	data.append(i)

label = []
for i in range(0,17):
	label.append(0.0)
label[10] = 1.0
v = sess.run(train_step, feed_dict={x: [data], y_:[label]})   
#v = sess.run(train_step)

#print v

