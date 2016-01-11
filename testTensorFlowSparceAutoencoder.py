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
n_lambda = 0.0001
n_sparcity = 0.01
n_beta = 3

learning_rate = .01

x = tf.placeholder("float", [None, n_input])
hidden = tf.placeholder("float", [None, n_hidden])

def autoencoder(X, weights, biases):
    hiddenlayer = tf.sigmoid(
        tf.add(
            tf.matmul(
                X, weights['hidden']
            ),
            biases['hidden']
        )
    )
    out = tf.matmul(hiddenlayer, weights['out']) + biases['out']
    return {'out': out, 'hidden': hiddenlayer}

weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}

#Construct my model
pred = autoencoder(x, weights, biases)

rho_hat = tf.reduce_mean(pred['hidden'])
sparce_cost = tf.sum(rho)

#Construct cost
cost = tf.add(tf.nn.l2_loss(pred-x) , tf.nn.l2_loss(weights.values())) # + sparcity






trainingX = gid.normalizeData(gid.getPatches())
W = np.random.random()





