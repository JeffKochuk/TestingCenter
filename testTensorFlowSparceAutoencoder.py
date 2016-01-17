import numpy as np
import matplotlib.pyplot as mp
import scipy.io
import visualizer as vis
import getImageData as gid
import tensorflow as tf
__author__ = 'JEFFERYK'

#Network Parameters
n_epoch_size = 100000
n_num_epochs = 2
n_input = 64
n_hidden = 25
n_output = n_input
n_lambda = tf.constant(0.0001)
n_rho = tf.constant(0.01)
n_beta = tf.constant(3.)
learning_rate = tf.constant(.01)
ONE = tf.constant(1.0)

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
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

#Construct my model
pred = autoencoder(x, weights, biases)

rho_hat = tf.reduce_mean(pred['hidden'],1)

def KL_Div(rho, rho_hat):
    invrho = tf.sub(tf.constant(1.), rho)
    invrhohat = tf.sub(tf.constant(1.), rho_hat)
    logrho = logfunc(rho,rho_hat) + logfunc(invrho, invrhohat)
    return logrho

def logfunc(x, x2):
    return tf.mul( x, tf.log(tf.div(x,x2)))

#Construct cost
cost_sparse = tf.mul(n_beta, tf.reduce_sum(KL_Div(n_rho, rho_hat)))
cost_J = tf.reduce_mean(tf.nn.l2_loss(tf.sub(pred['out'], x)))
cost_reg = tf.mul(n_lambda,tf.add(tf.nn.l2_loss(weights['hidden']),tf.nn.l2_loss(weights['out'])))
cost = tf.add(tf.add(cost_J , cost_reg ), cost_sparse)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)




# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()

    # Training cycle
    for epoch in range(n_num_epochs):
        avg_cost = 0.
        batch_xs=gid.getPatches(n_epoch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs})

    print("Optimization Finished!")

    saver.save(sess, 'my-SAE')

    outWeights = sess.run(weights['hidden'])
    outWeights = gid.normalizeData(outWeights)
    vis.display_network(outWeights)
    vis.display_network(outWeights.T, filename="weightsT.png")








