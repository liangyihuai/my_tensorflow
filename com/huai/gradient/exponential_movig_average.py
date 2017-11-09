# coding=
import tensorflow as tf;

v = tf.Variable(0, dtype=tf.float32);
step = tf.Variable(0, trainable=False);

exponential_moving_average = tf.train.ExponentialMovingAverage(0.99, step);
# 每次执行这个操作的时候列表中的变量就会更新
maintain_averages_op = exponential_moving_average.apply([v]);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());

    print(sess.run([v, exponential_moving_average.average(v)]));

    sess.run(tf.assign(v, 5)); # v = 5;
    sess.run(maintain_averages_op); # 更新变量
    # 0.1 * step + 0.9 * 5 = 4.5
    print(sess.run([v, exponential_moving_average.average(v)]));

    sess.run(tf.assign(step, 10000));
    sess.run(tf.assign(v, 10));
    sess.run(maintain_averages_op); # 更新变量
    #
    print(sess.run([v, exponential_moving_average.average(v)]))

    sess.run(maintain_averages_op); # 更新变量
    print(sess.run([v, exponential_moving_average.average(v)]))













