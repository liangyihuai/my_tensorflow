from numpy.random import RandomState;
import tensorflow as tf

# y = converlution_nerual_network.inference(x, False, None)
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

y = tf.Variable([1, 2, 1, 2]);
y_ = tf.Variable([1, 0, 1, 2]);

correct = tf.equal(y, y_)
cast_value = tf.cast(correct , tf.float32);
accuracy = tf.reduce_mean(cast_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    print(sess.run(correct));
    print(sess.run(cast_value));
    print(sess.run(accuracy));