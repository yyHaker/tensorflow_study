# -*- coding: utf-8 -*-
"""
use dropout layer:
tf.layers.dropout()
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# reproduce
tf.set_random_seed(1)
np.random.seed(1)

# Hyper parameters
N_SAMPLES = 20
N_HIDDEN = 300
LR = 0.01

# training data
x = np.linspace(-1, 1, N_SAMPLES)[:, np.newaxis]  # shape(20, 1)
y = x + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]    # add some noise data

# test data
test_x = x.copy()       # shape (20, 1)
test_y = test_x + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]

# show data
plt.scatter(x=x, y=y, c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(x=test_x, y=test_y, c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

# tf placeholders
tf_x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# to control dropout when training and testing
tf_is_training = tf.placeholder(dtype=tf.bool, shape=None)

# overfitting net
o1 = tf.layers.dense(inputs=tf_x, units=N_HIDDEN, activation=tf.nn.relu)
o2 = tf.layers.dense(inputs=o1, units=N_HIDDEN, activation=tf.nn.relu)
o_out = tf.layers.dense(inputs=o2, units=1)
# compute loss
o_loss = tf.losses.mean_squared_error(labels=tf_y, predictions=o_out)
o_train = tf.train.AdamOptimizer(LR).minimize(o_loss)

# dropout net
d1 = tf.layers.dense(inputs=tf_x, units=N_HIDDEN, activation=tf.nn.relu)
d1 = tf.layers.dropout(inputs=d1, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
d2 = tf.layers.dense(inputs=d1, units=N_HIDDEN, activation=tf.nn.relu)
d2 = tf.layers.dropout(inputs=d2, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
d_out = tf.layers.dense(inputs=d2, units=1)
# compute loss
d_loss = tf.losses.mean_squared_error(labels=tf_y, predictions=d_out)
d_train = tf.train.AdamOptimizer(LR).minimize(d_loss)

# create session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   # something about plotting

for t in range(500):
    # train
    sess.run([o_train, d_train], feed_dict={tf_x: x, tf_y: y, tf_is_training: True})  # train, set is_training=True

    if t % 10 == 0:
        # plotting
        plt.cla()  # clear the axis
        # test, set is_training=False
        o_loss_, d_loss_, o_out_, d_out_ = sess.run(
            [o_loss, d_loss, o_out, d_out], feed_dict={tf_x: test_x, tf_y: test_y, tf_is_training: False}
        )
        # plot the data
        plt.scatter(x=x, y=y, c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(x=test_x, y=test_y, c='cyan', s=50, alpha=0.3, label='test')

        plt.plot(test_x, o_out_, 'r-', lw=3, label='overfitting')
        plt.plot(test_x, d_out_, 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % o_loss_, fontdict={'size': 20, 'color':  'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % d_loss_, fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

plt.ioff()
plt.show()
