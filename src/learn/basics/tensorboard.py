# -*- coding: utf-8 -*-
import tensorflow as tf

with tf.name_scope('input1'):
    input1 = tf.constant([1.0,2.0,3.0],name="input1")
with tf.name_scope('input2'):
    input2 = tf.Variable(tf.random_uniform([3]),name="input2")
output = tf.add_n([input1,input2],name="add")

writer = tf.summary.FileWriter("D://logs", tf.get_default_graph())
writer.close()

# 1. 如果使用的是虚拟环境，请在虚拟环境下使用tensorboard 命令
# 2. logdir下的路径不需要全路径，相对路径也可以(网上的一些教程真的是坑)
