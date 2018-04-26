# -*- coding: utf-8 -*-
"""
1. 如果使用的是虚拟环境，请在虚拟环境下使用tensorboard 命令
2. logdir下的路径不需要全路径，相对路径也可以(网上的一些教程真的是坑)
-----
[1] tf.summary.histogram() 画直方图
[2] tf.summary.scalar()  画标量数据
"""
import tensorflow as tf
import numpy as np

# reproduce
tf.set_random_seed(1)
np.random.seed(1)

# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(loc=0, scale=0.1, size=x.shape)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise

with tf.variable_scope('Inputs'):
    tf_x = tf.placeholder(tf.float32, x.shape, name='x')
    tf_y = tf.placeholder(tf.float32, y.shape, name='y')

with tf.variable_scope('Net'):
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu, name='hidden_layer')
    output = tf.layers.dense(l1, 1, name='output_layer')

    # add to histogram summary (两层网络输出)
    tf.summary.histogram(name='h_out', values=l1)
    tf.summary.histogram(name='pred', values=output)

loss = tf.losses.mean_squared_error(labels=tf_y, predictions=output, scope='loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

# add loss to scalar summary
tf.summary.scalar(name='loss', tensor=loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('./log', sess.graph)     # write to file
merge_op = tf.summary.merge_all()                       # operation to merge all summary

for step in range(100):
    # train and net output
    _, result = sess.run([train_op, merge_op], feed_dict={tf_x: x, tf_y: y})
    # 每次都要写结果到tensorboard
    writer.add_summary(summary=result, global_step=step)

# Lastly, in your terminal or CMD, type this :
# $ tensorboard --logdir path/to/log
# open you google chrome, type the link shown on your terminal or CMD. (something like this: http://localhost:6006)


