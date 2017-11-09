import tensorflow as tf;

x = tf.get_variable('x', shape=[1], dtype='float32', initializer=tf.constant_initializer(5));

y = tf.square(x);

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(y);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    for i in range(10000):
        sess.run(train_op)
        if(i % 100) == 0:
            x_value = sess.run(x);
            print("iteration num: %s, the x value is: %f"%(i, x_value));