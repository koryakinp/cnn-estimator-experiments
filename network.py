import tensorflow as tf
import numpy as np


def get_model(x, conv, dense, apply_dropout=True):

    net = tf.reshape(x, [-1, 28, 28, 1])

    for i, val in enumerate(conv):
        net = tf.layers.conv2d(**build_conv_param(net, val, 'conv' + str(i)))
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.contrib.layers.flatten(net)

    for i, val in enumerate(dense[:-1]):
        net = tf.layers.dense(**build_dense_param(net, val, 'dense' + str(i)))
        net = tf.layers.dropout(net, rate=0.4, training=apply_dropout)

    return tf.layers.dense(**build_dense_param(net, dense[2], 'dense3'))


def build_conv_param(input, params, name):
    return {
      "inputs": input,
      "name": name,
      "filters": params["filters"],
      "kernel_size": params["filter_size"],
      "padding": 'valid',
      "activation": tf.nn.relu
    }


def build_dense_param(input, units, name):
    return {
      "inputs": input,
      "name": name,
      "units": units,
      "activation": tf.nn.relu
    }
