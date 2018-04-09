# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# creating layers
x = tf.placeholder(tf.float32, shape=[None, 3])
# outputs = activation(inputs * kernel + bias)
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
# initialize layers
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# execute layers
print(sess.run(y, feed_dict={x: [[1, 2, 3], [4, 5, 6]]}))

# tf.layers.dense  create and run the layer in a single call
