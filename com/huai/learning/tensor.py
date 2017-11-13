import tensorflow as tf;


def tensor_test():
    tensor = tf.get_variable('tensor', shape=[4], initializer=tf.random_normal_initializer(stddev=0.1))

    shape = tf.shape(tensor);

    v2 = tf.reshape(tensor, [-1, 2])
    v3 = tf.shape(v2)[0]

    v4 = tf.Print(tensor, [tensor])
    print(v4)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        value = sess.run(tensor)
        print(value);
        print(sess.run(shape));
        print(sess.run(v2));
        print(sess.run(v3));


def print_test():
    a = tf.constant([1.0, 4.0], shape=[2, 1])
    b = tf.constant([2.0, 3.0], shape=[1, 2])
    c = tf.add(tf.matmul(a, b), tf.constant([5.0, 6.0]))
    d = tf.Print(c, [c, 2.0], message="Value of C is:")
    with tf.Session() as sess:
        print(sess.run(d))


print_test();