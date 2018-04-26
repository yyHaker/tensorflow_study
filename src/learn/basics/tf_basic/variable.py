# -*- coding: utf-8 -*-
import tensorflow as tf

state = tf.Variable(0, name='counter')

# 定义常量one
one = tf.constant(1)

# 定义加法步骤
new_value = tf.add(state, one)

# 将state更新为new_value
update = tf.assign(state, new_value)

# 如果定义 Variable, 就一定要 initialize
init = tf.global_variables_initializer()
# 需要再在 sess 里, sess.run(init) , 激活 init 这一步
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
# 一定要把 sess 的指针指向 state 再进行 print 才能得到想要的结果
