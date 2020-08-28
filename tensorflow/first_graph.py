#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

# https://www.tensorflow.org/guide/effective_tf2

# Import mnist
mnist = tf.keras.datasets.mnist
(x_train, ytrain), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_tesT / 255.0


# Import mnist
mnist = tf.keras.datasets.mnist
(x_train, ytrain), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_tesT / 255.0


x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x * x * y + y + 2

with tf.Session() as ss:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)

