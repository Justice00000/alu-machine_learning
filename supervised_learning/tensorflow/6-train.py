#!/usr/bin/env python3
<<<<<<< HEAD
""" train"""
=======
"""
Defines a function that builds, trains, and saves
neural network classifier
"""
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6


import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


<<<<<<< HEAD
def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """ train model"""
=======
def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier

    parameters:
        X_train [numpy.ndarray]: contains training input data
        Y_train [numpy.ndarray]: contains training labels
        X_valid [numpy.adarray]: contains validation input data
        Y_valid [numpy.ndarray]: contains validation labels
        layer_sizes [list]: contains number of nodes in each layer of network
        activations [list]: contains activation functions for each layer
        alpha [float]: learning rate
        iterations [int]: number of iterations to train over
        save_path [string]: designates path for where to save the model

    returns:
        path to where model was saved
    """
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
<<<<<<< HEAD
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            cost_t = sess.run(loss,
                              feed_dict={x: X_train, y: Y_train})
            acc_t = sess.run(accuracy,
                             feed_dict={x: X_train, y: Y_train})
            cost_v = sess.run(loss,
                              feed_dict={x: X_valid, y: Y_valid})
            acc_v = sess.run(accuracy,
                             feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_t))
                print("\tTraining Accuracy: {}".format(acc_t))
                print("\tValidation Cost: {}".format(cost_v))
                print("\tValidation Accuracy: {}".format(acc_v))
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
=======

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations):
            loss_train = sess.run(loss,
                                  feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            loss_valid = sess.run(loss,
                                  feed_dict={x: X_valid, y: Y_valid})
            accuracy_valid = sess.run(accuracy,
                                      feed_dict={x: X_valid, y: Y_valid})
            if (i % 100) is 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(loss_train))
                print("\tTraining Accuracy: {}".format(accuracy_train))
                print("\tValidation Cost: {}".format(loss_valid))
                print("\tValidation Accuracy: {}".format(accuracy_valid))
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        i += 1
        loss_train = sess.run(loss,
                              feed_dict={x: X_train, y: Y_train})
        accuracy_train = sess.run(accuracy,
                                  feed_dict={x: X_train, y: Y_train})
        loss_valid = sess.run(loss,
                              feed_dict={x: X_valid, y: Y_valid})
        accuracy_valid = sess.run(accuracy,
                                  feed_dict={x: X_valid, y: Y_valid})
        print("After {} iterations:".format(i))
        print("\tTraining Cost: {}".format(loss_train))
        print("\tTraining Accuracy: {}".format(accuracy_train))
        print("\tValidation Cost: {}".format(loss_valid))
        print("\tValidation Accuracy: {}".format(accuracy_valid))
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
        return saver.save(sess, save_path)
