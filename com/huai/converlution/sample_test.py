import tensorflow as tf;
import numpy as np;

M = np.array([
    [[1], [-1], [0]],
    [[-1], [2], [1]],
    [[0], [2], [-2]]
])

print ('Matrix shape is:', M.shape);
filter_weight = tf.get_variable('weights', [2, 2, 1, 1],
                                initializer=tf.constant_initializer(
                                    [[1, -1],
                                    [0, 2]]))
biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(1))

M = np.asarray(M, dtype='float32');
M = M.reshape(1, 3, 3, 1);

print("M: ", M);


x = tf.placeholder('float32', [1, None, None, 1]);
conv = tf.nn.conv2d(x, filter_weight, strides=[1, 2, 2, 1], padding='SAME');
bias = tf.nn.bias_add(conv, biases);
pool = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');
with tf.Session() as sess:
    tf.global_variables_initializer().run();
    convolution_M = sess.run(bias, feed_dict={x:M});
    pool_M = sess.run(pool, feed_dict={x:M});

    print("convolution M: ", convolution_M);
    print("pooled M: ", pool_M);



















