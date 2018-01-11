"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

import tempfile

from com.huai.converlution.resnets.resnets_utils import *

import tensorflow as tf


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 64, 64, 3])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 3, 32])
        # b_conv1 = bias_variable([32])

        training = tf.placeholder(tf.bool)
        conv1 = conv2d(x_image, W_conv1)
        batch_norm1 = tf.layers.batch_normalization(conv1, axis=3, training=training)

        # h_conv1 = tf.nn.relu(batch_norm1 + b_conv1)
        h_conv1 = tf.nn.relu(batch_norm1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        # b_conv2 = bias_variable([64])
        conv2 = conv2d(h_pool1, W_conv2)
        batch_norm2 = tf.layers.batch_normalization(conv2, axis=3, training=training)
        h_conv2 = tf.nn.relu(batch_norm2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([16 * 16 * 64, 1024])
        b_fc1 = bias_variable([1, 1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 6])
        b_fc2 = bias_variable([1, 6])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob, training


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Create the model
    x = tf.placeholder(tf.float32, [None, 64, 64, 3])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None, 6])

    # Build the graph for the deep net
    y_conv, keep_prob, is_training = deepnn(x)

    with tf.name_scope('loss'):
        # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    orig_data = load_dataset('D:/LiangYiHuai/deepleanring/resnets/datasets')
    X_train, Y_train, X_test, Y_test = process_orig_datasets(orig_data)
    mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=32, seed=None)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            X_mini_batch, Y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]
            if i % 20 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: X_mini_batch,
                                                          y_: Y_mini_batch, keep_prob: 1.0, is_training: False})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: X_mini_batch,
                                      y_: Y_mini_batch, keep_prob: 0.5, is_training: True})

        print('test accuracy %g' % accuracy.eval(feed_dict={x: X_test,
                                                            y_: Y_test, keep_prob: 1.0, is_training: False}))


if __name__ == '__main__':
    tf.app.run(main=main)
