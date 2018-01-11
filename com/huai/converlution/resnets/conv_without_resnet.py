from com.huai.converlution.resnets.resnets_utils import *
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def reference(features, model):
    conv1 = tf.layers.conv2d(features, filters=64, kernel_size=[5, 5], padding='valid',
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[5, 5],
                             padding='valid', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.layers.flatten(pool2)

    dense1 = tf.layers.dense(pool2_flat, units=512, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(dense1, training=(model == tf.estimator.ModeKeys.TRAIN))
    dense2 = tf.layers.dense(dropout1, units=256, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(dense2, training=(model==tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs=dropout2, units=6)
    return logits


features = tf.placeholder(name='feature', dtype=tf.float32, shape=(None, 64, 64, 3))
labels = tf.placeholder(name='label', dtype=tf.float32, shape=(None, 6))

logits = reference(features, tf.estimator.ModeKeys.TRAIN)

loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

optimizer = tf.train.AdamOptimizer(learning_rate=0.005)

train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1)), tf.float32))

orig_data = load_dataset('D:/LiangYiHuai/deepleanring/resnets/datasets')
X_train, Y_train, X_test, Y_test = process_orig_datasets(orig_data)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=16, seed=None)

    for i in range(1000):

        X_mini_batch, Y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]

        x_to_feed = X_mini_batch

        y_to_feed = Y_mini_batch

        _, loss_sess, accuracy_sess = sess.run([train_op, loss, accuracy_op], feed_dict={features:x_to_feed, labels:y_to_feed})

        if i % 30 == 0:
            print(i, loss_sess, accuracy_sess)

