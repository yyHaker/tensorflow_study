# -*- coding: utf-8 -*-
import os
from data.cifar10 import Corpus

from model.basic_rnn import ConvNet
cifar10 = Corpus()


def basic_rnn():
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24)
    convnet.train(dataloader=cifar10, backup_path='backup/cifar10-v14/',
                  batch_size=128, n_epoch=500)


basic_rnn()
