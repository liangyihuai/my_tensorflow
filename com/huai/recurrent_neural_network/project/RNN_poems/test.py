import tensorflow as tf;
import numpy as np;

c = np.random.random([10, 1])
b = tf.nn.embedding_lookup(c, [1, 3, 4])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(b))
    print(c)








