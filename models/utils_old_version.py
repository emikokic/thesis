# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:49:41 2018
@author: Ferdinand
"""

import tensorflow as tf
import numpy as np


def flat(x, output_name="flatten"):
    """ takes a tensor of rank > 2 and return a tensor of shape [?,n]"""
    shape = x.get_shape().as_list()
    n = np.prod(shape[1:])
    x_flat = tf.reshape(x, [-1, n])
    return tf.identity(x_flat, name=output_name)


def max_pool(x, layer_spec, output_name="max_pool"):
    """max_pool downsamples a feature map by specification on layer_spec."""
    return tf.nn.max_pool(x, ksize=[1, 1, layer_spec["pool_size"], 1],
                          strides=[1, 1, layer_spec["pool_size"], 1],
                          padding='SAME', name=output_name)


def unpool(value, layer_spec, output_shape, name='unpool'):
    """
    N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
    """
    with tf.name_scope(name):
        kernel_shape = (1, layer_spec["pool_size"])
        num_channels = value.get_shape()[-1]
        input_dtype_as_numpy = value.dtype.as_numpy_dtype()
        kernel_rows, kernel_cols = kernel_shape

        # build kernel
        kernel_value = np.zeros((kernel_rows, kernel_cols,
                                 num_channels, num_channels),
                                dtype=input_dtype_as_numpy)
        kernel_value[:, :, :, :] = np.eye(num_channels, num_channels)
        kernel = tf.constant(kernel_value)

        # do the un-pooling using conv2d_transpose
        out = tf.nn.conv2d_transpose(value,
                                     kernel,
                                     output_shape=output_shape,
                                     strides=(1, kernel_rows, kernel_cols, 1),
                                     padding='VALID')

    return out


def run_layer(x, layer_spec, output_name):
    """ x: a tensor
        layer_spec: a dictionary containing the type of the layer (dense or conv2d) 
                    and the shape of the kernel ([a,b] for dense layer, [a,b,c,d] for conv2d
        return the output of the layer"""
    if layer_spec["type"] == "dense":
        W = tf.get_variable("W", layer_spec["shape"],
                            initializer=tf.random_normal_initializer())
        return tf.matmul(x, W, name=output_name)
    elif layer_spec["type"] == "conv":
        filters = tf.get_variable("W", (layer_spec["kernel_size"], 1,
                                        layer_spec["input_filters"],
                                        layer_spec["output_filters"]),
                                  initializer=tf.random_normal_initializer())

        print('input shape', x.shape)
        print('filter shape', filters.shape)
        return tf.nn.conv2d(x, filters, strides=[1, 1, 1, 1], padding='SAME',
                            name=output_name)


def run_transpose_layer(x, layer_spec, output_name="inverse", output_shape=None):
    """ Performs the transposed operation of the layer described by layer_spec.
        x: a tensor whose shape is the same as the output of the layer
        layer_spec: dictionary describing the layer
        output shape: shape of the output tensor
        return a tensor whose shape is the same as the input of the layer
        """
    if layer_spec["type"] == "dense":
        W = tf.get_variable("W", layer_spec["shape"][::-1],
                            initializer=tf.random_normal_initializer())
        return tf.matmul(x, W, name=output_name)

    elif layer_spec["type"] == "conv":
        kernel_shape = (layer_spec["kernel_size"], 1,
                        layer_spec["output_filters"],
                        layer_spec["input_filters"])
        filters = tf.get_variable("W", kernel_shape,
                                  initializer=tf.random_normal_initializer())
        return tf.nn.conv2d(x, filters, strides=[1, 1, 1, 1], padding='SAME',
                            name=output_name)

    elif layer_spec["type"] == "flat":
        return tf.reshape(x, output_shape, name=output_name)

    elif layer_spec["type"] == "max_pool":
        return unpool(x, layer_spec, output_shape, name=output_name)
