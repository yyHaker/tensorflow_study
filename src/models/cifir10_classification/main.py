# -*- coding: utf-8 -*-
import os
from data.cifar10 import Corpus
from data.DataImage import DataImage

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
    train_settings.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    train_settings.add_argument("--adjust_learning_rate", type=bool, default=False, help="if adjust learning rate")
    train_settings.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    train_settings.add_argument("--batch_normal", type=bool, default=True, help="if use batch normalization")
    train_settings.add_argument("--resp_normal", type=bool, default=False, help="if use local resp_normal")
    train_settings.add_argument("--dropout_keep_prob", type=float, default=0.5, help='dropout keep prob')
    train_settings.add_argument("--batch_size", type=int, default=16, help='train batch size')
    train_settings.add_argument("--epochs", type=int, default=100, help='train epochs')

    model_settings = parse.add_argument_group("model settings")
    model_settings.add_argument("--n_channels", type=int, default=3, help="the image input channels")
    model_settings.add_argument("--n_classes", type=int, default=20, help="the image classes")
    model_settings.add_argument("--image_size", type=int, default=224, help="the image size")

    path_settings = parse.add_argument_group("path settings")
    path_settings.add_argument("--backup_path", default="backup/cifar10-v14/", help="the model saved directory")
    path_settings.add_argument("--log_path", default="backup/logs/train.log", help="path to log the file...")

    return parse.parse_args()


def train(args):
    """train the classification model"""
    logger = logging.getLogger("image classification")
    logger.info("get the image data.....")
    # cifar10 = Corpus()
    data_image = DataImage()
    logger.info("initialize the model...")
    convnet = ConvNet(args)
    logger.info("training the model....")
    convnet.train(dataloader=data_image, backup_path=args.backup_path,
                  batch_size=args.batch_size, n_epoch=args.epochs)
    logger.info("done with the training!")


def run():
    """run the system"""
    args = parse_args()

    logger = logging.getLogger("image classification")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    if args.log_path:  # logging 不会自己创建目录，但是会自己创建文件
        log_dir = os.path.dirname(args.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # 默认输出到console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Running with args: {}".format(args))
    if args.train:
        train(args)


if __name__ == "__main__":
    run()



