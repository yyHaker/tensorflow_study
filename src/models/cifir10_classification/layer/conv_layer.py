# -*- coding: utf-8 -*-
import numpy
import tensorflow as tf


class ConvLayer:
    """a convolution layer."""
    def __init__(self, input_shape, n_size, n_filters, stride=1, activation='relu',
                 batch_normal=False, weight_decay=None, name='conv'):
        """
        :param input_shape: the input rgb image data, (batch, height, width, channel).
        :param n_size: the filter size (height=width)
        :param n_filters: the number of filters
        :param stride:
        :param activation:
        :param batch_normal:
        :param weight_decay:
        :param name:
        """
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.activation = activation
        self.stride = stride
        self.batch_normal = batch_normal
        self.weight_decay = weight_decay

        # 权重矩阵
        self.weight = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[n_size, n_size, self.input_shape[3], self.n_filters],
                mean=0.0,
                stddev=2.0 / (self.input_shape[1] * self.input_shape[2] * self.input_shape[3])
            ),
            name='W_%s' % name
        )

        # weight decay
        if self.weight_decay:
            weight_decay = tf.multiply(tf.nn.l2_loss(self.weight), self.weight_decay)
            tf.add_to_collection('losses', weight_decay)

        # 偏置向量
        self.bias = tf.Variable(initial_value=tf.constant(0.0, shape=[self.n_filters]),
                                name='b_%s' % name)

        # batch_normalization
        if self.batch_normal:
            self.epsilon = 1e-5
            self.gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[self.n_filters]),
                                     name='gamma_%s' % name)

    def get_output(self, input):
        # caculate input_shape and output_shape
        self.output_shape = [self.input_shape[0], int(self.input_shape[1]/self.stride),
                             int(self.input_shape[2]/self.stride), self.n_filters]

        # hidden states    # input [batch, in_height, in_width, in_channels]
        self.conv = tf.nn.conv2d(input=input, filter=self.weight,
                                 strides=[1, self.stride, self.stride, 1],
                                 padding='SAME')

        # batch normalization          # weight [filter_size, filter_size, in_channels, out_channels]
        if self.batch_normal:           # 此处的self.conv的维度是? [batch, in_height, in_width, in_channels]?
            mean, variance = tf.nn.moments(self.conv, axes=[0, 1, 2], keep_dims=False)
            self.hidden = tf.nn.batch_normalization(self.conv, mean=mean, variance=variance,
                                                    offset=self.bias, scale=self.gamma, variance_epsilon=self.epsilon)
        else:
            self.hidden = self.conv + self.bias

        # activation
        if self.activation == 'relu':
            self.output = tf.nn.relu(self.hidden)
        elif self.activation == 'tanh':
            self.output = tf.nn.tanh(self.hidden)
        elif self.activation == 'sigmoid':
            self.output = tf.nn.sigmoid(self.hidden)
        elif self.activation == 'none':
            self.output = self.hidden
        return self.output

