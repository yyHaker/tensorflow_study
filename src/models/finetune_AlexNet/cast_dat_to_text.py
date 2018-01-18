# -*- coding: utf-8 -*-
"""
transfer the zip data to the train.txt and val.txt
"""
import os
import numpy as np
train_file = 'path/to/train.txt'
val_file = 'path/to/val.txt'
test_file = 'path/to/test.txt'

transfer_file = "data/train/"

# 将指定的路径下的images路径写到文件中
filenames = os.listdir(transfer_file)
print(len(filenames))
np.random.shuffle(filenames)

# split the data 7:3
# write to train.txt
with open(train_file, "w") as f:
    for names in filenames[:2500*7]:
        if "cat" in names:
            names = names + " 0"
        else:
            names = names + " 1"
        print(transfer_file + names)
        f.writelines(transfer_file + names + "\n")

# write to val.txt
with open(val_file, "w") as f:
    for names in filenames[2500*7:]:
        if "cat" in names:
            names = names + " 0"
        else:
            names = names + " 1"
        f.writelines(transfer_file + names + "\n")
