'''
Save and Restore a model using TensorFlow.
This example is using the MNIST database of handwritten digits

do_sess1() and do_sess2() are almost the same. in do_sess2(), the weights and biases are used,
which are trained by do_sess1();'''

from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os;

# Import MNIST data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = "C:\\Users\\liangyh\\Desktop\\temp\\temp_model3\\mnist_model"

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


def forward_propagation(x):
    with tf.variable_scope('layer1'):
        hidden_layer1_weights = tf.get_variable('weights',
                                  shape=[n_input, n_hidden_1], initializer=tf.random_normal_initializer(stddev=0.1))
        hidden_layer1_biases = tf.get_variable('biases',
                                 shape=[n_hidden_1], initializer=tf.constant_initializer(0.1));

        # Hidden layer with RELU activation
        layer1 = tf.add(tf.matmul(x, hidden_layer1_weights), hidden_layer1_biases)
        layer1 = tf.nn.relu(layer1)
    with tf.variable_scope('layer2'):
        hidden_layer2_weights = tf.get_variable('weights',
                                                 shape=[n_hidden_1, n_hidden_2], initializer=tf.random_normal_initializer(stddev=0.1))
        hidden_layer2_biases = tf.get_variable('biases',
                                               shape=[n_hidden_2], initializer=tf.constant_initializer(0.1))
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer1, hidden_layer2_weights), hidden_layer2_biases)
        layer_2 = tf.nn.relu(layer_2)
    with tf.variable_scope('out-layer'):
        out_layer_weights = tf.get_variable('weights',
                                            shape=[n_hidden_2, n_classes], initializer=tf.random_normal_initializer(stddev=0.1))
        out_layer_biases = tf.get_variable('biases',
                                           shape=[n_classes], initializer=tf.constant_initializer(0.1))
    out_layer = tf.matmul(layer_2, out_layer_weights) + out_layer_biases;
    return out_layer


def do_sess1():
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Construct model
    pred = forward_propagation(x)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

    # Running first session
    print("Starting 1st session...")
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training cycle
        for epoch in range(3):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("First Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        # Save model weights to disk
        save_path = saver.save(sess, model_path);
        print("Model saved in file: %s" % save_path)


def do_sess2():
    # tf.reset_default_graph();
    with tf.Graph().as_default() as g:
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])

        # Construct model
        pred = forward_propagation(x)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

        # Running first session
        print("Starting 1st session...")
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            # Restore model weights from previously saved model
            saver.restore(sess, model_path)
            print("Model restored from file: %s" % model_path)

            # Training cycle
            for epoch in range(7):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print("First Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


do_sess1();
do_sess2();
