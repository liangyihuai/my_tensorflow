import tensorflow as tf;
import numpy as np;
import threading;
import time;


def my_loop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print('stoping from id: %d\n' % worker_id);
            coord.request_stop();
        else: print('working on id: %d\n'%worker_id);

        time.sleep(1);

def coord_test():
    coord = tf.train.Coordinator();
    threads = [threading.Thread(target=my_loop, args=(coord, i))for i in range(5)]
    for t in threads: t.start();

    coord.join(threads)


def queue_runner_test():
    queue = tf.FIFOQueue(100, 'float');
    enqueue_op = queue.enqueue([tf.random_normal([1])]);

    qr = tf.train.QueueRunner(queue, [enqueue_op] * 5);

    tf.train.add_queue_runner(qr);
    out_tensor = queue.dequeue();
    with tf.Session() as sess:
        coord = tf.train.Coordinator();
        threads = tf.train.start_queue_runners(sess=sess, coord=coord);
        for _ in range(3): print(sess.run(out_tensor)[0]);

        coord.request_stop();
        coord.join(threads);


queue_runner_test();









