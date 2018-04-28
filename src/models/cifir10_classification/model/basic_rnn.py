# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging

from layer.conv_layer import ConvLayer
from layer.dense_layer import DenseLayer
from layer.pool_layer import PoolLayer


class ConvNet(object):
    """the basic cnn models"""
    def __init__(self, args):
        # logging
        self.logger = logging.getLogger("image classification")

        # basic config params
        self.n_channels = args.n_channels
        self.n_classes = args.n_classes
        self.image_size = args.image_size

        self.optim = args.optim
        self.learning_rate = args.learning_rate
        self.adjust_learning_rate = args.adjust_learning_rate
        self.weight_decay = args.weight_decay
        self.batch_normal = args.batch_normal
        self.resp_normal = args.resp_normal
        self.dropout_keep_prob = args.dropout_keep_prob
        self.batch_size = args.batch_size
        self.epochs = args.epochs

        # input variable
        self.images = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size,
                                                              self.image_size, self.n_channels], name='images')
        self.labels = tf.placeholder(dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        self.global_step = tf.Variable(initial_value=0, dtype=tf.int32, name='global_step')

        # Network
        conv_layer1 = ConvLayer(input_shape=(None, self.image_size, self.image_size, self.n_channels),
                                n_size=3, n_filters=64, stride=1, activation='relu', batch_normal=self.batch_normal,
                                weight_decay=self.weight_decay, name='conv1')
        pool_layer1 = PoolLayer(n_size=4, stride=4, mode='max', resp_normal=self.resp_normal, name='pool1')

        conv_layer2 = ConvLayer(input_shape=(None, int(self.image_size/4), int(self.image_size/4), 64),
                                n_size=3, n_filters=128, stride=1, activation='relu', batch_normal=self.batch_normal,
                                weight_decay=self.weight_decay, name='conv2')
        pool_layer2 = PoolLayer(n_size=4, stride=4, mode='max', resp_normal=self.resp_normal, name='pool2')

        conv_layer3 = ConvLayer(input_shape=(None, int(self.image_size/16), int(self.image_size/16), 128),
                                n_size=3, n_filters=256, stride=1, activation='relu', batch_normal=self.batch_normal,
                                weight_decay=self.weight_decay, name='conv3')
        pool_layer3 = PoolLayer(n_size=4, stride=4, mode='max', resp_normal=self.resp_normal, name='pool3')

        dense_layer1 = DenseLayer(input_shape=(None, int(self.image_size/64)*int(self.image_size/64)*256),
                                  hidden_dim=4096, activation='relu', dropout=True, keep_prob=self.keep_prob,
                                  batch_normal=self.batch_normal, weight_decay=self.weight_decay, name='dense1')

        dense_layer2 = DenseLayer(input_shape=(None, 4096),
                                  hidden_dim=self.n_classes, activation='none', dropout=False, keep_prob=None,
                                  batch_normal=False, weight_decay=self.weight_decay, name='dense2')

        # data flow
        hidden_conv1 = conv_layer1.get_output(input=self.images)
        hidden_pool1 = pool_layer1.get_output(input=hidden_conv1)
        hidden_conv2 = conv_layer2.get_output(input=hidden_pool1)
        hidden_pool2 = pool_layer2.get_output(input=hidden_conv2)
        hidden_conv3 = conv_layer3.get_output(input=hidden_pool2)
        hidden_pool3 = pool_layer3.get_output(hidden_conv3)

        input_dense1 = tf.reshape(hidden_pool3, shape=[-1, int(self.image_size/64)*int(self.image_size/64)*256])
        output_dense1 = dense_layer1.get_output(input=input_dense1)
        logits = dense_layer2.get_output(input=output_dense1)

        # cross entropy object function
        self.objective = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=logits))
        tf.add_to_collection('losses', self.objective)

        self.avg_loss = tf.add_n(tf.get_collection('losses'))

        # learning rate
        if self.adjust_learning_rate:
            lr = tf.cond(tf.less(self.global_step, 50000),
                         lambda: tf.constant(0.01),
                         lambda: tf.cond(tf.less(self.global_step, 100000),
                                         lambda: tf.constant(0.001),
                                         lambda: tf.constant(0.0001))
                         )
        else:
            lr = self.learning_rate
        # optimizer
        if self.optim == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        elif self.optim == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif self.optim == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
        elif self.optim == "rprop":
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        else:
            raise NotImplementedError("unsupported optimizer: {}".format(self.optim))
        self.train_op = self.optimizer.minimize(self.avg_loss, global_step=self.global_step)

        # prediction
        correct_prediction = tf.equal(self.labels, tf.argmax(logits, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype='float'))

        self.logger.info('Build the network done....')

    def train(self, dataloader, backup_path, n_epoch=5, batch_size=128):
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
            self.logger.info("create backup directory....")
        self.logger.info("open the session.........")
        # build the session
        # 设置每个GPU应该拿出多少容量给进程使用
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # model saver
        self.saver = tf.train.Saver(var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=10)
        # model initialize
        self.sess.run(tf.global_variables_initializer())

        # training
        self.logger.info("begin training........")
        best_dev_accuracy = 0.0
        for epoch in range(0, n_epoch+1):
            # load data and data augmentation
            train_images = dataloader.data_augmentation(dataloader.train_images, mode='train',
                                                        flip=False, crop=False, whiten=False,
                                                        noise=False)
            train_labels = dataloader.train_labels
            valid_images = dataloader.data_augmentation(dataloader.valid_images, mode='test',
                                                        flip=False, crop=False, whiten=False,
                                                        noise=False)
            valid_labels = dataloader.valid_labels
            # print("compute train loss....")
            train_loss = 0.0
            for i in range(0, dataloader.n_train, batch_size):
                batch_images = train_images[i: i+batch_size]
                batch_labels = train_labels[i: i+batch_size]
                [_, avg_loss, iteration] = self.sess.run(fetches=[self.train_op, self.avg_loss, self.global_step],
                                                         feed_dict={self.images: batch_images,
                                                                    self.labels: batch_labels,
                                                                    self.keep_prob: self.dropout_keep_prob})
                train_loss += avg_loss * batch_images.shape[0]   # self.avg_loss计算的不是一个batch的损失吗?
            train_loss = 1.0 * train_loss / dataloader.n_train

            # get the loss and accuracy of the valid dataset
            # print("get the loss and accuracy of the valid dataset on epoch %d......" % epoch)
            valid_accuracy, valid_loss = 0.0, 0.0
            for i in range(0, dataloader.n_valid, batch_size):
                batch_images = valid_images[i: i+batch_size]
                batch_labels = valid_labels[i: i+batch_size]
                [avg_accuracy, avg_loss] = self.sess.run(
                    fetches=[self.accuracy, self.avg_loss],
                    feed_dict={
                        self.images: batch_images,
                        self.labels: batch_labels,
                        self.keep_prob: 1.0
                    }
                )
                valid_accuracy += avg_accuracy * batch_images.shape[0]
                valid_loss += avg_loss * batch_images.shape[0]
            valid_accuracy = 1.0 * valid_accuracy / dataloader.n_valid
            valid_loss = 1.0 * valid_loss / dataloader.n_valid

            self.logger.info("epoch{%d}, iter[%d], train loss: %.6f, valid precision: %.6f, "
                             "valid loss: %.6f" % (epoch, iteration, train_loss, valid_accuracy, valid_loss))
            sys.stdout.flush()

            # save the best model
            if valid_accuracy > best_dev_accuracy:
                saver_path = self.saver.save(self.sess, os.path.join(backup_path, 'model_%d.ckpt' % epoch))
                best_dev_accuracy = valid_accuracy
                self.logger.info("saved the best model of valid precision: %.6f" % valid_accuracy)

        self.sess.close()

    def test(self, dataloader, backup_path, epoch, batch_size=128):
        # 设置每个GPU应该拿出多少容量给进程使用
        gpu_options = tf.GPUOptions(per_process_gpu_memery=0.25)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # load the model
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.v2)
        model_path = os.path.join(backup_path, 'model_%d.ckpt' % epoch)
        assert os.path.exists(model_path + ".index")
        self.saver.restore(self.sess, model_path)
        self.logger.info("read model from %s" % model_path)

        # test the model
        accuarcy_list = []
        test_images = dataloader.data_augmentation(dataloader.test_images, flip=False, crop=True,
                                                   crop_shape=(24, 24, 3), whiten=True, noise=False)
        test_labels = dataloader.test_labels
        for i in range(0, dataloader.n_test, batch_size):
            batch_images = test_images[i: i+batch_size]
            batch_labels = test_labels[i: i+batch_size]
            [avg_accuracy] = self.sess.run(
                fetches=[self.accuracy],
                feed_dict={
                    self.images: batch_images,
                    self.labels: batch_labels,
                    self.keep_prob: 1.0
                }
            )
            accuarcy_list.append(avg_accuracy)
        self.logger.info("the model on the test precision: %.4f" % np.mean(accuarcy_list))
        self.sess.close()


