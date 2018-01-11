import tensorflow as tf

x = tf.Variable(0)
new_x = tf.assign(x, 1)

b = tf.Variable(initial_value=True, dtype=tf.bool)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(new_x)
    print(sess.run(b))

    sess.run(tf.assign(b, False))


    print(sess.run(b))
    print(sess.run(x))



