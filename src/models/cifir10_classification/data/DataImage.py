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


class DataImage(object):
    """the image data object"""
    def __init__(self):
        self.logger = logging.getLogger("image classification")
        self.load_image_data('image_scene_data/data')
        self._split_train_valid(valid_rate=0.9)
        # self.n_train = self.train_images.shape[0]
        # self.n_valid = self.valid_images.shape[0]
        # self.n_test = self.test_images.shape[0]

    def load_image_data(self, data_dir):
        """
        load image data
        :param data_dir: teh image data dir
        :return:
        """
        total_image = 0
        images = []
        for filename in os.listdir(data_dir)[40]:
            # print(filename)
            img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_COLOR)
            # real = cv2.resize(img, (1024, 768))
            # show image
            real = img
            cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(filename, real)
            print(real.shape)
            images.append(real)
            total_image += 1
            if total_image % 5 == 0:
                cv2.destroyAllWindows()
        print("total  images: ", total_image)
        cv2.destroyAllWindows()

    def _split_train_valid(self, valid_rate=0.9):
        pass


if __name__ == "__main__":
    image_data = DataImage()
