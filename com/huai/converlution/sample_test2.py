import tensorflow as tf;
import numpy as np;

input = np.array([[[1], [-1], [0]],
          [[-1], [2], [1]],
          [[0], [2], [-2]]])
input = np.asarray(input, dtype='float32')
input = input.reshape([1, 3, 3, 1]);

print("input shape: ", input.shape);
weights = tf.get_variable('weights', shape=[2, 2, 1, 1],
                         initializer=tf.constant_initializer([[1, -1], [0, 2]]))
biases = tf.get_variable('biases', shape=[1], dtype='float32', initializer=tf.constant_initializer(1));

x = tf.placeholder(dtype='float32', shape=[1, None, None, 1]);
conv = tf.nn.conv2d(x, weights, strides=[1, 2, 2, 1], padding='SAME');
bias = tf.nn.bias_add(conv, biases);
pool = tf.nn.avg_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');

with tf.Session() as sess:
    tf.global_variables_initializer().run();
    convoluted_M = sess.run(bias, feed_dict={x:input});
    pooled_M = sess.run(pool, feed_dict={x:input});

    print("convolution: ", convoluted_M);
    print("pool: ", pooled_M);

