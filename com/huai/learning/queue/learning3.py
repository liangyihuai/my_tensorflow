# coding=utf8
import tensorflow as tf
import time

"""
说明：QueueRunner的例子有一个问题：由于入队线程自顾自地执行，
在需要的出队操作完成之后，程序没法结束。
使用tf.train.Coordinator来终止其他线程。
其实可以认为是做一些线程间的同步关系。
在QueueRunner中，increment_op和enqueue_op两个操作在不同的子线程当中。
"""

q = tf.FIFOQueue(1000, 'float')
counter = tf.Variable(0.0)
increment_op = tf.assign_add(counter, tf.constant(1.0))
enqueue_op = q.enqueue(counter)

qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op]*1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

    time.sleep(2)

    print('len', sess.run(q.size()))
    # main thread
    for i in range(0, 10):
        print('-------------')
        print(sess.run(q.dequeue()))
        print('len', sess.run(q.size()))
    # inform other threads to stop
    coord.request_stop()
    # main thread waits until other threads stop.
    coord.join(enqueue_threads)


