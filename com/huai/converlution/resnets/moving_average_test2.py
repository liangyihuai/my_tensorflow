import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

_extra_train_ops = []

mean = tf.Variable(initial_value=2, dtype=tf.float32)
variance = tf.Variable(initial_value=2, dtype=tf.float32)

moving_mean = tf.get_variable(
                    'moving_mean', tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
moving_variance = tf.get_variable(
    'moving_variance', tf.float32,
    initializer=tf.constant_initializer(1.0, tf.float32),
    trainable=False)

_extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
_extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))

train_op = tf.group(*_extra_train_ops)

# for vars in tf.all_variables():
#     print(vars.name)
#
with tf.variable_scope('moving_mean', reuse=True):
    moving_biased = tf.get_variable('biased')
    moving_local_step = tf.get_variable('local_step')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    sess.run(train_op)
    print(sess.run([moving_mean, moving_variance]))

    sess.run(tf.assign(mean, 4))
    sess.run(tf.assign(variance,6))
    sess.run(train_op)
    print(sess.run([moving_mean, moving_variance]))

    sess.run(tf.assign(mean, 6))
    sess.run(tf.assign(variance, 8))
    sess.run(train_op)
    print(sess.run([moving_mean, moving_variance]))

    # for i in range(30):
    #     sess.run(train_op)
    #     print(sess.run([moving_mean, moving_variance]))



























