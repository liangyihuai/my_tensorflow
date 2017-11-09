# coding=
import tensorflow as tf;

'''逐渐减少learning rate'''

global_step = tf.Variable(0);
# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
learning_rate = tf.train.exponential_decay(1.0, global_step=global_step,
                                           decay_steps=1, decay_rate=0.96, staircase=True);

x = tf.get_variable('x', shape=[1], dtype='float32', initializer=tf.constant_initializer(5));

y = tf.square(x);

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(y,global_step=global_step);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    for i in range(100):
        sess.run(train_op)
        if(i % 10) == 0:
            x_value = sess.run(x);
            print("iteration num: %s, the x value is: %f"%(i, x_value));