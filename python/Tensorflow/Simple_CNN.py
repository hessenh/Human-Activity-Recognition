import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data	
import input_data_window_large

class Simple_network(object):
	def __init__(self, data_set, output_size, model):
		super(Simple_network, self).__init__()
	    
	    # Setting the data set
		self.data_set = data_set

	    # Input, bias and weights
		self.x = tf.placeholder(tf.float32, [None, 900])
		self.W = tf.Variable(tf.zeros([900, output_size]), name = model + 'W')
		self.b = tf.Variable(tf.zeros([output_size]), name = model + 'b')
	    
		self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

	    # Output prediction
		self.y_ = tf.placeholder(tf.float32, [None, output_size])

		self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))


		self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)

		self.init = tf.initialize_all_variables()

	def save_variables(self, model):
		saver = tf.train.Saver()
		

		#save_path = saver.save(model_vars, model)
		save_path = saver.save(self.sess, model)
		print("Model saved in file: %s" % save_path)

	def load_model(self, model):
		self.sess = tf.Session()
		all_vars = tf.all_variables()
		model_vars = [k for k in all_vars if k.name.startswith(model)]
		tf.train.Saver(model_vars).restore(self.sess, model)

	def train_network(self):
		self.sess = tf.Session()
		self.sess.run(self.init)
		for i in range(1000):
		  batch_xs, batch_ys = self.data_set.train.next_batch(100)
		  self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

	def test_network(self):
		correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(self.sess.run(accuracy, feed_dict={self.x: self.data_set.test.data, self.y_: self.data_set.test.labels}))

train_subjects = ["P03","P04"]#,"P06","P07","P08","P09","P10","P14","P15","P16","P17","P18","P19","P20","P21"]
test_subjects = ["P11"]#,"P04","P06","P07","P08","P09","P10","P14","P15","P16","P17","P18","P19","P20","P21"]
subject_set = [train_subjects, test_subjects]
convertion = {1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:2, 17:2}
number_of_labels = len(set(convertion.values()))
data_set_1 = input_data_window_large.read_data_sets(subject_set, number_of_labels, convertion, None)

#data_set_1 = input_data.read_data_sets("MNIST_data/", one_hot=True)

nn = Simple_network(data_set_1,2,'model_1')
#nn.train_network()
#nn.test_network()
#nn.save_variables("model_1")
nn.load_model('model_1')
nn.test_network()


convertion_2 = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}
number_label_2 = len(set(convertion_2.values()))
data_set_2 = input_data_window_large.read_data_sets(subject_set, number_label_2, convertion_2, None)

nn2 = Simple_network(data_set_2, 17, 'model_2')
#nn2.train_network()
#nn2.test_network()
#nn2.save_variables("model_2")
nn2.load_model('model_2')
nn2.test_network()
