# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# define inputs and expected output
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

# build a simple linear model with 1 output
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

# initiate global variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print("current pred:", sess.run(y_pred))

# define mean square error to optimise the model
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print("current loss:", sess.run(loss))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)


# training 100 epoch
for i in range(100):
    _, loss_value = sess.run((train, loss))
    print("epoch: ", i, " ", loss_value)

print(sess.run(y_pred))