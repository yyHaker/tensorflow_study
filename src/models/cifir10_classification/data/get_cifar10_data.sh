#!/usr/bin/env bash
# get the data from the web and extract to CIFAR10_data1
if [[-d CIFAR10_data]]; then
    echo "data exist"
    exit 0
else
    wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -zxvf cifar-10-python.tar.gz
    mv cifar-10-batches-py CIFAR10_data
    echo "data is done, saved to CIFAR10_data"
fi
