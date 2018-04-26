# -*- coding: utf-8 -*-
import os
from data.cifar10 import Corpus

from model.basic_rnn import ConvNet

import argparse
import logging


def parse_args():
    """parses command lines"""
    parse = argparse.ArgumentParser("classification on the image data")
    parse.add_argument("--train", action='store_true', help='train the model')
    parse.add_argument("--test", action='store_true', help="test the model")

    train_settings = parse.add_argument_group("train settings")
    train_settings.add_argument("--optim", default="adam", help='optimizer type')
    train_settings.add_argument("--learning_rate", type=float, default=0.001)
    train_settings.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    train_settings.add_argument("--dropout_keep_prob", type=float, default=0.5, help='dropout keep prob')
    train_settings.add_argument("--batch_size", type=int, default=128, help='train batch size')
    train_settings.add_argument("--epochs", type=int, default=10, help='train epochs')

    model_settings = parse.add_argument_group("model settings")
    model_settings.add_argument("--n_channels", type=int, default=3, help="the image input channels")
    model_settings.add_argument("--n_classes", type=int, default=10, help="the image classes")
    model_settings.add_argument("--image_size", type=int, default=24, help="the image size")

    path_settings = parse.add_argument_group("path settings")
    path_settings.add_argument("--backup_path", default="backup/cifar10-v14/", help="the model saved directory")
    path_settings.add_argument("--log_path", default="backup/logs/logs.log", help="path to log the file...")

    return parse.parse_args()


def train(args):
    """train the classification model"""
    logger = logging.getLogger("image classification")
    logger.info("get the image data.....")
    cifar10 = Corpus()
    logging.info("initialize the model...")
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24)
    logging.info("training the model....")
    convnet.train(dataloader=cifar10, backup_path='backup/cifar10-v14/',
                  batch_size=128, n_epoch=500)
    logging.info("done with the training!")


def run():
    """run the system"""
    args = parse_args()

    logger = logging.getLogger("image classification")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # 默认输出到console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info("Running with args: {}".format(args))
    if args.train:
        train(args)


