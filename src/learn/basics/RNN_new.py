# -*- codingL: utf-8 -*-
"""
learn to use RNN.
others:
tf.reshape()
TODO: how to realize RNN GRU LSTM and bi-directional  ?
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

# reproduce
tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate

# data
mnist = input_data.read_data_sets(train_dir='./MNIST_data', one_hot=True)    # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
# show an image
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()

# tensorflow placeholders
tf_x = tf.placeholder(dtype=tf.float32, shape=[None, TIME_STEP * INPUT_SIZE])       # shape(batch, 784)
image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])          # shape(batch, height, width)
tf_y = tf.placeholder(dtype=tf.int32, shape=[None, 10])          # input y

# build the network: a RNN layer + linear layer
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    cell=rnn_cell,                   # cell you have chosen
    inputs=image,                      # input
    initial_state=None,         # the initial hidden state  (how to initialize?)
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
# add a neural layer
output = tf.layers.dense(inputs=outputs[:, -1, :], units=10)              # output based on the last output step

# compute cost
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

# return (acc, update_op), and create 2 local variables
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))[1]

# create session
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

# begin training
for step in range(1200):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], feed_dict={tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:      # testing
        accuracy_ = sess.run(accuracy, feed_dict={tf_x: test_x, tf_y: test_y})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

# print 10 predictions from test data
test_output = sess.run(output, feed_dict={tf_x: test_x[:10]})
pred_y = np.argmax(test_output, axis=1)    # use the max value to predict
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], axis=1), 'real number')
