import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
import numpy as np

words_in_dataset = np.array(10*[np.ones(10)])

#print words_in_dataset
lstm_size = 10
batch_size = 5

lstm = rnn_cell.BasicLSTMCell(lstm_size)
print lstm.__dict__
# Initial state of the LSTM memory.
state = tf.zeros([batch_size, lstm.state_size])

loss = 0.0
for current_batch_of_words in words_in_dataset:
	#print current_batch_of_words
	# The value of state is updated after processing each batch of words.
	output, state = lstm(current_batch_of_words, state)

	# The LSTM output can be used to make next word predictions
	logits = tf.matmul(output, softmax_w) + softmax_b
	probabilities = tf.nn.softmax(logits)
	loss += loss_function(probabilities, target_words)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
