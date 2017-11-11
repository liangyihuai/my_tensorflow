import tensorflow as tf;
import numpy as np;
from tensorflow.examples.tutorials.mnist import input_data;
import os;
import time


INPUT_NODE = 784;
OUTPUT_NODE = 10;

IMAGE_SIZE = 28;
NUM_CHANNELS = 1;
NUM_LABELS = 10;

CONV1_DEEP = 32;
CONV1_SIZE = 5;

CONV2_DEEP = 64;
CONV2_SIZE = 5;

FC_SIZE = 512;

BATCH_SIZE = 100;
LEARNING_RATE_BASE = 0.1;
LEARNING_RATE_DECAY = 0.99;
REGULARIZATION_RATE = 0.0001;
TRAINING_STEPS = 6000;
MOVING_AVERAGE_DECAY = 0.99;

MODEL_SAVE_PATH = 'C:\\Users\\USER\\Desktop\\model_saving_path';
MODEL_NAME = "mnist_model"

def inference(input_tensor, train, regularizer):
    with tf.variable_scope("layer1_conv1"):
        conv1_weights = tf.get_variable(
            'weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev = 0.1))
        conv1_biases = tf.get_variable(
            'bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0));
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME");
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases));

    with tf.name_scope("layer2_pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME");

    with tf.variable_scope("layer3_conv2"):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0));
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME');
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases));

    with tf.name_scope("layer4_pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME");
        pool_shape = pool2.get_shape().as_list();
        nodes = pool_shape[1]*pool_shape[2] * pool_shape[3];
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes]);

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights));
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1));

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases);
        if train: fc1 = tf.nn.dropout(fc1, 0.5);

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight', [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1));
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights));
        fc2_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights)+fc2_biases;

    return logit;


def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS],name='x-input');
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input');
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE);
    y = inference(x, False, regularizer);
    global_step = tf.Variable(0, trainable=False);

    variable_averages = tf.train.ExponentialMovingAverage(LEARNING_RATE_DECAY, global_step);
    variable_averages_op = variable_averages.apply(tf.trainable_variables());

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy);
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'));

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples/BATCH_SIZE,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step);
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train');

    saver = tf.train.Saver();
    with tf.Session() as sess:
        tf.global_variables_initializer().run();
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE);

            reshaped_xs = np.reshape(xs, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS));
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:reshaped_xs, y_:ys})

            if i % 1000 == 0:
                print("after %d training step(s), loss on training batch is %g."%(step, loss_value));
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step);


def main():
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True);
    train(mnist)
    # evaluate(mnist)

if __name__ == '__main__':
    main();










