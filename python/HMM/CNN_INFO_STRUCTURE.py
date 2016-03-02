''' Used to understand the CNN structure'''

import tensorflow as tf

#x = tf.Variable([1,2,3,4,5,6,7,8,9])
x = tf.placeholder("float", shape=[1, 576])

''' Forste lag '''
# w
w_1 = tf.truncated_normal([5,5,1,32], stddev=0.1)
w_1 =  tf.Variable(w_1)
# b
b_1 = tf.constant(0.1, shape=[32])
b_1 = tf.Variable(b_1)

# Data points
x_image = tf.reshape(x, [-1, 24, 24, 1])

h_1 = tf.nn.conv2d(x_image, w_1, strides=[1, 1, 1, 1], padding='SAME')
h_1 = tf.nn.relu(h_1 + b_1)
h_p1 = tf.nn.max_pool(h_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

''' Andre lag'''
w_2 = tf.truncated_normal([5,5,32,64], stddev=0.1)
w_2 = tf.Variable(w_2)
#print w_2.get_shape()
b_2 = tf.constant(0.1, shape=[64])
b_2 = tf.Variable(b_2)
h_2 = tf.nn.conv2d(h_p1, w_2, strides=[1, 1, 1, 1], padding='SAME')
#print h_2.get_shape()
h_2 = tf.nn.relu(h_2 + b_2)
h_p2 = tf.nn.max_pool(h_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print h_p2.get_shape()
''' NN '''
w_fc1 = tf.truncated_normal([6*6*64,1024])
w_fc1 = tf.Variable(w_fc1)
#print w_fc1.get_shape()
b_fc1 = tf.constant(0.1, shape=[1024])
b_fc1 = tf.Variable(b_fc1)
#print b_fc1.get_shape()

h_p2_flat = tf.reshape(h_p2, [-1, 6*6*64])
print h_p2_flat.get_shape()
h_fc1 = tf.matmul(h_p2_flat, w_fc1)
#print h_fc1.get_shape()
h_fc1 = tf.nn.relu(h_fc1 + b_fc1)

w_fc2 = tf.truncated_normal([1024,17])
w_fc2 = tf.Variable(w_fc2)
#print w_fc2.get_shape()
b_fc2 = tf.constant(0.1, shape=[17])
b_fc2 = tf.Variable(b_fc2)
#print b_fc2.get_shape()

y_conv = tf.matmul(h_fc1, w_fc2)
y_conv = tf.nn.softmax(y_conv + b_fc2)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
data = []
for i in range(1,577):
	data.append(i)

v = sess.run(w_1)
#print v[0][0][0][0]
v = sess.run(x_image, feed_dict={x: [data]})   
#print v[0][0][0][0]
v = sess.run(h_1, feed_dict={x: [data]})
#print v[0][1][0]
v = sess.run(h_p1, feed_dict={x: [data]})
#print len(v[0][0][0])
v = sess.run(w_2)
#print v[0][0][0]
v = sess.run(b_2)
#print v
v = sess.run(h_2, feed_dict={x: [data]})
#print v[0][0][0]
v = sess.run(h_p2, feed_dict={x: [data]})
#print v[0][0][0]
v = sess.run(w_fc1)
#print len(v)
v = sess.run(h_p2_flat, feed_dict={x:[data]})
#print v
v = sess.run(h_fc1, feed_dict={x:[data]})
#print v
v = sess.run(w_fc2)
#print len(v)
v = sess.run(y_conv, feed_dict={x:[data]})
print v