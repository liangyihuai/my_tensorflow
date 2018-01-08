import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE_NUMBER = 784
OUTPUT_NODE_NUMBER = 10

LEARNING_RATE = 0.05
EPOCHES_NUMBER = 1000

x = tf.placeholder(shape=(None, 784), dtype=tf.float32, name='x')
y = tf.placeholder(shape=(None, 10), dtype=tf.int8, name='y')

w1 = tf.get_variable('w1', (INPUT_NODE_NUMBER, 1000), initializer=tf.random_normal_initializer)
b1 = tf.zeros((1, 1000), name='b1')
a1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.get_variable('w2', (1000, OUTPUT_NODE_NUMBER), initializer=tf.random_normal_initializer)
b2 = tf.zeros((1, OUTPUT_NODE_NUMBER), name='b2')
z2 = tf.matmul(a1, w2) + b2

cross_entropy_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=z2))

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mnist = input_data.read_data_sets('F:/PycharmProject/TensorFlowTest/com/huai/converlution/MNIST-data', one_hot=True)

    for i in range(EPOCHES_NUMBER):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size=100)
        _, cost = sess.run([train_op, cross_entropy_cost], feed_dict={x:batch_xs, y:batch_ys})

        if i % 50 == 0:
            print("cost %f" % cost)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(z2, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = sess.run(accuracy, feed_dict={x: mnist.validation.images, y :mnist.validation.labels})
    print("accuracy: %s"%accuracy)
















