import tensorflow as tf

q = tf.FIFOQueue(6, tf.float32)

init = q.enqueue_many(([0.1, 0.2, 0.3, 7, 6, 5], ))

x = q.dequeue()
y = x + 1
z = q.enqueue([y])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(init)

    sess.run(z)
    sess.run(z)

    len = sess.run(q.size())

    print('len', len)

    for i in range(len):
        print(sess.run(q.dequeue()))


