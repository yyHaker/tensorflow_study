# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

"""
DataSet API
"""
my_data =[
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7]
]
# create DataSet according to the given tensor
slices = tf.data.Dataset.from_tensor_slices(my_data)
# create iterator
next_item = slices.make_one_shot_iterator().get_next()
# read data until there is no more data
# create session to run
sess = tf.Session()
while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break

# If the Dataset depends on stateful operations you may need to initialize
# the iterator before using it
r = tf.random_normal([10, 3])
dataset = tf.data.Dataset.from_tensor_slices(r)
# create iterator(need initialize)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()
# initialize this iterator
sess.run(iterator.initializer)
while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break