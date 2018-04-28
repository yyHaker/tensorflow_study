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
    """the image data object

    vocabs below:
    self.class_to_id:
    self.id_to_class:
    self.imgname_to_classid:

    """
    def __init__(self, image_width=768, image_height=768, images_dir='data/image_scene_data/data',
                 labels_path='data/image_scene_data/list.csv',
                 categories_path='data/image_scene_data/categories.csv'):
        # resize the image to what size
        self.image_width = image_width
        self.image_height = image_height
        self.logger = logging.getLogger("image classification")
        self.build_class_vocab(categories_path)
        self.load_image_data(images_dir, labels_path)
        self.analyze_image_data()
        self._split_train_valid(valid_rate=0.9)
        self.n_train = self.train_images.shape[0]
        self.n_valid = self.valid_images.shape[0]
        # self.n_test = self.test_images.shape[0]

    def analyze_image_data(self):
        images_labels = self.images_labels
        class_count = {}
        labels = []
        for image_label in images_labels:
            labels.append(image_label[1])
        lbl_set = set(labels)
        for lbl in lbl_set:
            class_count[self.id_to_class[lbl]] = labels.count(lbl)

        self.logger.info("The training image data infor: {}".format(class_count))

    def build_class_vocab(self, categories_path):
        """
        build id_to_class and class_to_id vocab..
        :param categories_path:
        :return:
        """
        self.logger.info("build id_to_class and class_to_id vocab....")
        categories_csv = csv.reader(open(categories_path, 'r', encoding='utf-8'))
        id_to_class = {}
        class_to_id = {}
        for c in categories_csv:
            if "ID" in c:
                continue
            id_to_class[c[0]] = c[2]
        for k, v in id_to_class.items():
            class_to_id[v] = k
        self.class_to_id, self.id_to_class = class_to_id, id_to_class
        self.logger.info("build class vocab done")
        self.logger.info("id_to_class: {}".format(id_to_class))
        self.logger.info("class_to_id: {}".format(class_to_id))

    def load_image_data(self, images_dir, labels_path):
        """
        load image data, and build vocabs.
        :param images_dir: teh image data dir
        :return:
        """
        # read labels
        self.logger.info("read labels....")
        self.imgname_to_classid = {}
        list_csv = csv.reader(open(labels_path, 'r'))
        for img in list_csv:
            if "FILE_ID" in img:
                continue
            self.imgname_to_classid[img[0]] = img[1]
        # read and show images
        self.logger.info("read images....")
        images_labels = []
        count = 0
        for filename in os.listdir(images_dir):
            img = cv2.imread(os.path.join(images_dir, filename))
            real = cv2.resize(img, (self.image_width, self.image_height))  # numpy array
            count += 1
            if count % 1000 == 0:
                self.logger.info("read images: {}".format(count))
            # show image (get the image label)
            image_name = filename.split('.')[0]
            class_id = self.imgname_to_classid[image_name]
            label_name = self.id_to_class[class_id]
            images_labels.append((real, class_id))
            # cv2.namedWindow(label_name)
            # cv2.imshow(label_name, real)
            # cv2.waitKey(0)
            if count % 5 == 0:
                cv2.destroyAllWindows()
        cv2.destroyAllWindows()
        self.logger.info("total read images and labels: {}".format(count))
        # conver to numpy array
        self.images_labels = np.array(images_labels)

    def _split_train_valid(self, valid_rate=0.9):
        images_labels = self.images_labels
        # shuffle
        np.random.shuffle(images_labels)
        # split the data
        thresh = int(valid_rate * self.images_labels.shape[0])
        self.train_images, self.train_labels = [], []
        self.valid_images, self.valid_labels = [], []
        for img, lbl in images_labels[: thresh]:
            self.train_images.append(img)
            self.train_labels.append(lbl)
        for img, lbl in images_labels[thresh:]:
            self.valid_images.append(img)
            self.valid_labels.append(lbl)
        self.train_images, self.train_labels = np.array(self.train_images
                                                        , dtype=float), np.array(self.train_labels, dtype=int)
        self.valid_images, self.valid_labels = np.array(self.valid_images
                                                        ,dtype=float), np.array(self.valid_labels, dtype=int)
        self.logger.info("split data result: train images{}, valid images: {}".format(
            self.train_images.shape[0], self.valid_images.shape[0]))

    def data_augmentation(self, images, mode='train', flip=False,
                          crop=False, crop_shape=(32, 32, 3), whiten=False,
                          noise=False, noise_mean=0, noise_std=0.01):
        """
        data augmentation.
        :param images:
        :param mode:
        :param flip:
        :param crop: if crop the image
        :param crop_shape: crop a shape of the image
        :param whiten:
        :param noise:
        :param noise_mean:
        :param noise_std:
        :return:
        """
        if not crop_shape:
            crop_shape = (self.image_width, self.image_height)
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
    logger = logging.getLogger("image classification")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    # 默认输出到console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    image_data = DataImage(images_dir='image_scene_data/data',
                           labels_path='image_scene_data/list.csv',
                           categories_path='image_scene_data/categories.csv')
