import numpy as np
import matplotlib.pyplot as mp
import scipy.io
import getImageData as gid
import tensorflow as tf
__author__ = 'JEFFERYK'

#Network Parameters
n_input = 64
n_hidden = 25
n_output = n_input

learning_rate = .01

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_input])

def autoencoder(X, weights, biases):
    hiddenlayer = tf.sigmoid(
        tf.add(
            tf.matmul(
                X, weights['hidden']
            ),
            biases['hidden']
        )
    )
    return tf.matmul(hiddenlayer, weights['out']) + biases['out']

weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}





trainingX = gid.normalizeData(gid.getPatches())
W = np.random.random()
for row in train:
   train(row)





