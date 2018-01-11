import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('C:/Users/USER/Desktop/temp/MNIST-data', one_hot=True, seed=1)


def reference(features, model):
    conv1 = tf.layers.conv2d(features, filters=32, kernel_size=[5, 5], padding='valid',
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
                             padding='valid', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.layers.flatten(pool2)

    dense1 = tf.layers.dense(pool2_flat, units=512, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(dense1, training=(model == tf.estimator.ModeKeys.TRAIN))
    dense2 = tf.layers.dense(dropout1, units=256, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(dense2, training=(model==tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs=dropout2, units=10)
    return logits


features = tf.placeholder(name='feature', dtype=tf.float32, shape=(None, 28, 28, 1))
labels = tf.placeholder(name='label', dtype=tf.float32, shape=(None, 10))

logits = reference(features, tf.estimator.ModeKeys.TRAIN)
cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
train_op = tf.train.AdamOptimizer(0.002).minimize(cross_entropy_loss)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    x, y = mnist.train.next_batch(128)
    x_to_feed = np.reshape(x, (-1, 28, 28, 1))
    y_to_feed = y
    for i in range(10000):
        _, accuracy, loss = sess.run([train_op, accuracy_op, cross_entropy_loss], feed_dict={features:x_to_feed, labels:y_to_feed})

        if i % 200 == 0:
            print(i, loss, accuracy)

    train_x = np.reshape(mnist.train.images, (-1, 28, 28, 1))
    train_y = mnist.train.labels
    train_accuracy_value = sess.run(accuracy_op, feed_dict={features: train_x, labels:train_y})
    print('training accuracy: %f'%train_accuracy_value)

    validation_x = np.reshape(mnist.validation.images, (-1, 28, 28, 1))
    validation_y = mnist.validation.labels
    accuracy_value = sess.run(accuracy_op, feed_dict={features: validation_x, labels: validation_y})
    print("validation accuracy: %f"%accuracy_value)




