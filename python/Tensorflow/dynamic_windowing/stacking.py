import tensorflow as tf
import numpy as np
import pandas as pd
filepath = 'original_5_classifiers.csv'
df = pd.read_csv(filepath, header=None, sep='\,',engine='python')
print df.iloc[0]
l = 395182
print l
data = np.zeros([l,5*17])



label = np.zeros([l,17])


# Shuffle data
perm = np.arange(l)
np.random.shuffle(perm)
data = data[perm]
label = label[perm]


x = tf.placeholder(tf.float32, [None, 85])


W = tf.Variable(tf.zeros([85, 17]))
b = tf.Variable(tf.zeros([17]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 17])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)

sample = 0
i = 0
iteration = 0
start = l*2/3
stop = len(df)
while(iteration < 1000):
	if (i*100+100) >= start:
		i=0
	batch_xs = data[i*100:i*100+100]
	batch_ys = label[i*100:i*100+100]
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	i+=1
	iteration +=1

print(sess.run(accuracy, feed_dict={x: data[start:stop], y_: label[start:stop]}))























