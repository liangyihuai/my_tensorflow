import tensorflow as tf
import sys
import time

q = tf.FIFOQueue(1000, 'float')
counter = tf.Variable(0.0)
increment_op = tf.assign_add(counter, tf.constant(1.0))
enqueue_op = q.enqueue(counter)

qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # start another thread to "increment_op" and "enqueue_op";
    enqueue_threads = qr.create_threads(sess, start=True)
    print('len = ', sess.run(q.size()))
    # main thread to dequeue
    for i in range(10):
        print('--------------------')
        print(sess.run(q.dequeue()))








