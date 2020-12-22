#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TensorFlow Low-Level API

import tensorflow as tf
import numpy as np

# Simple Math Tensor (Array)
tensor_1 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Shape and DType
print(f"Shape: {tensor_1.shape}")
print(f"DType: {tensor_1.dtype}")

# Indexing
index = tensor_1[:, 1:]
# Ellipsis operator - https://stackoverflow.com/questions/772124/what-does-the-ellipsis-object-do
wtf = tensor_1[..., 1, tf.newaxis]

# Tensor Operations
tensor2 = tensor_1 + 10
tensor3 = tensor2 - 10
tensor4 = tensor3 * 100
tensor5 = tensor4 / 10
# tensor_1.__add__(10) equivalent

tensor_squared = tensor_1.tensor_squared(tensor_1)

# Transpose tensor
tensor_transposed = tf.transpose(tensor_1)

# Matrix Mult tf.matmult()
tensor_matrix_multiply = tensor_1 @ tensor_transposed

tensor_squeezed = tf.squeeze(wtf)

# reducing
tensor_reduced_mean = tf.reduce_mean(tensor_1)
tensor_reduced_sum = tf.reduce_sum(tensor_1)
tensor_reduced_max = tf.reduce_max(tensor_1)

# NumPy 32 bit and Tensorflow 32bit
numpy_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
tensor_from_numpy = tf.constant(numpy_array)

numpy_array64 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
tensor_from_numpy64 = tf.constant(numpy_array64)

# Compue the "kleines einmaleins"
new_matrix = tf.reshape(tensor_from_numpy, (-1, 1)) @ tf.transpose(
    tf.reshape(tf.cast(tensor_from_numpy64, dtype=tf.float32), (-1, 1))
)

# in-memory operations using tf.Variable
v = tf.Variable([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v.assign(2 * v)
v[1, 1].assign(999)
v.scatter_nd_update(indices=[[0, 0], [2, 2]], updates=[333, 666])

