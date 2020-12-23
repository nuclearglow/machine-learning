#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 23:18:32 2020

@author: plkn
"""

# In tf
v = tf.Variable(np.random.randint(low=-10, high=10, size=(20, 20)), dtype=tf.int32)
v_np = v.numpy()


nonzero_idx = tf.cast(tf.not_equal(v, 0), tf.int32)
nonzero_idx_np = nonzero_idx.numpy()

# Count True entries
n_nonzero = tf.reduce_sum(nonzero_idx)
n_nonzero_np = n_nonzero.numpy()

# Buils a vector of ones
ones_vec = tf.ones((n_nonzero), dtype=tf.int32)
ones_vec_np = ones_vec.numpy()

v.scatter_nd_update(indices=nonzero_idx, updates=ones_vec)
                             
       
                             
# In numpy
w = np.random.randint(low=-10, high=10, size=(20, 20))   

idx = w == 7              

w[idx]= 999     