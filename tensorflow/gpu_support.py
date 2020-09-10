#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# HOWTO:
# First Install Tensorflow 2 with GPU support:
# conda install tensorflow-gpu
# Link: https://medium.com/@dkinghorn/tensorflow-2-gpu-with-anaconda-python-no-separate-cuda-install-needed-d10cddb444b1

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
