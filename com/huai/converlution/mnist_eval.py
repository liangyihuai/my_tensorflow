# coding=utf8

import tensorflow as tf;
import numpy as np;
from tensorflow.examples.tutorials.mnist import input_data;
import os;
import time
from com.huai.converlution import converlution_nerual_network

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # x = tf.placeholder(tf.float32, [None, converlution_nerual_network.INPUT_NODE], name='x-input')
        x = tf.placeholder(tf.float32, [None, converlution_nerual_network.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, converlution_nerual_network.OUTPUT_NODE], name='y-input')

        reshaped_x= np.reshape(mnist.validation.images,
                               (len(mnist.validation.labels),
                                converlution_nerual_network.IMAGE_SIZE,
                                converlution_nerual_network.IMAGE_SIZE,
                                converlution_nerual_network.NUM_CHANNELS))

        validate_feed = {x: reshaped_x, y_: mnist.validation.labels}

        y = converlution_nerual_network.inference(x, False, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(converlution_nerual_network.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(converlution_nerual_network.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)



def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()