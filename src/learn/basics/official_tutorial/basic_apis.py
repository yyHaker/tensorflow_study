# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# create two floating point constants a and b
# build the graph
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0, dtype=tf.float32)
total = a + b
print(a)
print(b)
print(total)

# visualizing a computation graph
writer = tf.summary.FileWriter('logdir')
writer.add_graph(tf.get_default_graph())

# create session to run
sess = tf.Session()
print(sess.run(total))

# run can handle any combination of tuples or dictionaries
print(sess.run({'ab': (a, b), 'total': total}))

# generate a random 3-element vector in [0, 1)
vec = tf.random_uniform(shape=(3, ))
print(sess.run(vec))

# Feeding
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
# feed data to run
# note(the feed_dict argument can be used to overwrite any tensor in the graph)
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))





