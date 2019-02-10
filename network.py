import tensorflow as tf
import numpy as np


def get_model(x, apply_dropout=True):

    net = tf.reshape(x, [-1, 28, 28, 1])

    net = tf.layers.conv2d(
        inputs=net,
        name="conv1",
        filters=48,
        kernel_size=5,
        padding="valid",
        activation=tf.nn.relu)

    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(
        inputs=net,
        name="conv2",
        filters=48,
        kernel_size=5,
        padding="valid",
        activation=tf.nn.relu)

    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(net, name="dense1", units=256, activation=tf.nn.relu)
    return tf.layers.dense(inputs=net, name="output", units=10)
