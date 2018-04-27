# -*- coding:utf-8 -*-
"""
read the image data from the data file
"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import os
import cv2
import logging
import csv
import random

# reproduce
np.random.seed(1)


class DataImage(object):
    """the image data object"""
    def __init__(self):
        self.logger = logging.getLogger("image classification")
        self.load_image_data('data/image_scene_data/data', 'data/image_scene_data/list.csv')
        self._split_train_valid(valid_rate=0.9)
        self.n_train = self.train_images.shape[0]
        self.n_valid = self.valid_images.shape[0]
        # self.n_test = self.test_images.shape[0]

    def load_image_data(self, images_dir, labels_path):
        """
        load image data
        :param images_dir: teh image data dir
        :return:
        """
        # read and show images
        self.logger.info("read images....")
        images = []
        count = 0
        for filename in os.listdir(images_dir):
            img = cv2.imread(os.path.join(images_dir, filename))
            real = cv2.resize(img, (32, 32))  # numpy array
            count += 1
            if count % 1000 == 0:
                self.logger.info("read images: {}".format(count))
            # show image
            # real = img
            # cv2.namedWindow(filename)
            # cv2.imshow(filename, real)
            # print(real.shape)
            images.append(real)
            # if total_image % 5 == 0:
                # cv2.destroyAllWindows()
        # cv2.destroyAllWindows()
        self.logger.info("total read images: {}".format(count))
        # read labels
        self.logger.info("read labels....")
        labels = []
        list_csv = csv.reader(open(labels_path, 'r'))
        for img in list_csv:
            if "FILE_ID" in img:
                continue
            labels.append(img[1])
        # get the train data(images and labels)
        self.images = np.array(images, dtype=float)
        self.labels = np.array(labels)
        assert self.images.shape[0] == self.labels.shape[0], \
            "images nums: {} not equal to labels num: {}".format(
                self.images.shape[0], self.labels.shape[0])
        self.logger.info("load images: {}, load labels:{}".format(self.images.shape[0],
                                                                  self.labels.shape[0]))

    def _split_train_valid(self, valid_rate=0.9):
        images, labels = self.images, self.labels
        image_label = []
        for image, label in zip(images, labels):
            image_label.append((image, label))
        image_label = np.array(image_label)
        self.logger.info("the length of image_label: ", len(image_label))
        # shuffle
        np.random.shuffle(image_label)
        # split the data
        thresh = int(valid_rate * self.images.shape[0])
        self.train_images, self.train_labels = [], []
        self.valid_images, self.valid_labels = [], []
        for img, lbl in image_label[: thresh]:
            self.train_images.append(img)
            self.train_labels.append(lbl)
        for img, lbl in image_label[thresh:]:
            self.valid_images.append(img)
            self.valid_labels.append(lbl)
        self.train_images, self.train_labels = np.array(self.train_images
                                                        ), np.array(self.train_labels)
        self.valid_images, self.valid_labels = np.array(self.valid_images
                                                        ), np.array(self.valid_labels)
        self.logger.info("split data result: train images{}, valid images: {}".format(
            self.train_images.shape[0], self.valid_images.shape[0]))

    def data_augmentation(self, images, mode='train', flip=False,
                          crop=False, crop_shape=(32, 32, 3), whiten=False,
                          noise=False, noise_mean=0, noise_std=0.01):
        # 图像切割
        if crop:
            if mode == 'train':
                images = self._image_crop(images, shape=crop_shape)
            elif mode == 'test':
                images = self._image_crop_test(images, shape=crop_shape)
        # 图像翻转
        if flip:
            images = self._image_flip(images)
        # 图像白化
        if whiten:
            images = self._image_whitening(images)
        # 图像噪声
        if noise:
            images = self._image_noise(images, mean=noise_mean, std=noise_std)

        return images

    def _image_crop(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            left = np.random.randint(old_image.shape[0] - shape[0] + 1)
            top = np.random.randint(old_image.shape[1] - shape[1] + 1)
            new_image = old_image[left: left + shape[0], top: top + shape[1], :]
            new_images.append(new_image)

        return np.array(new_images)

    def _image_crop_test(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            left = int((old_image.shape[0] - shape[0]) / 2)
            top = int((old_image.shape[1] - shape[1]) / 2)
            new_image = old_image[left: left + shape[0], top: top + shape[1], :]
            new_images.append(new_image)

        return np.array(new_images)

    def _image_flip(self, images):
        # 图像翻转
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            if np.random.random() < 0.5:
                new_image = cv2.flip(old_image, 1)
            else:
                new_image = old_image
            images[i, :, :, :] = new_image
        return images

    def _image_whitening(self, images):
        # 图像白化
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            new_image = (old_image - np.mean(old_image)) / np.std(old_image)
            images[i, :, :, :] = new_image

        return images

    def _image_noise(self, images, mean=0, std=0.01):
        # 图像噪声
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            new_image = old_image
            for i in range(images.shape[0]):
                for j in range(images.shape[1]):
                    for k in range(images.shape[2]):
                        new_image[i, j, k] += random.gauss(mean, std)
            images[i, :, :, :] = new_image
        return images


if __name__ == "__main__":
    image_data = DataImage()
