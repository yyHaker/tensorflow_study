# -*- coding: utf-8 -*-
import tensorflow as tf

# placeholder 是 Tensorflow 中的占位符，暂时储存变量.
# 在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
x1 = tf.placeholder(dtype=tf.float32, shape=None)
y1 = tf.placeholder(dtype=tf.float32, shape=None)
# mul = multiply 是将input1和input2 做乘法运算，并输出为 output
z1 = tf.multiply(x1, y1)

x2 = tf.placeholder(dtype=tf.float32, shape=[2, 1])
y2 = tf.placeholder(dtype=tf.float32, shape=[1, 2])
z2 = tf.matmul(x2, y2)   # 矩阵乘法

with tf.Session() as sess:
    # when only one operation to run
    z1_value = sess.run(z1, feed_dict={x1: 1, y1: 2})

    # when run multiple operations
    # run can handle any combination of tuples or dictionaries
    z1_value, z2_value = sess.run([z1, z2], feed_dict={
        x1: 1, y1: 2,
        x2: [[2], [2]], y2: [[3, 3]]
    })

print(z1_value)
print(z2_value)
