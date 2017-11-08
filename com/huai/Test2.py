import tensorflow as tf

g1 = tf.Graph();
with g1.as_default():
    v = tf.get_variable("v", [1], initializer=tf.zeros_initializer());

g2 = tf.Graph();
with g2.as_default():
    v = tf.get_variable("v", [1], initializer=tf.ones_initializer());

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run();
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")));

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")));
















