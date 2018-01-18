# -*- coding: utf-8 -*-
import numpy as np
a = {"a": 3, "b": 7, "i": 90}
# print(a.items())

# load pretrained weights
weights_path = "bvlc_alexnet.npy"
weights_dict = np.load(weights_path, encoding='bytes').item()
print(weights_dict)  # a dict of lists
print("-"*80)
print(weights_dict["fc6"])
print("-"*80)
for opt_name in weights_dict:
    print(opt_name)

