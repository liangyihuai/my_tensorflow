import tensorflow as tf

x = tf.Variable(0)
new_x = tf.assign(x, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(new_x)
    print(sess.run(x))