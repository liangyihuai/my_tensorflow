import tensorflow as tf;

q = tf.FIFOQueue(2, 'int32');

init = q.enqueue_many([[0, 10], ]);
x = q.dequeue();
y = x + 1;

q_inc = q.enqueue([y]);

with tf.Session() as sess:
    init.run(); # init the queue
    for _ in range(5):
        v, _ = sess.run([x, q_inc]);
        print(v);



