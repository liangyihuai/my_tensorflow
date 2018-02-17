import numpy as np
import tensorflow as tf


# data I/O
# should be simple plain text file
data = open('C:/Users/USER/Desktop/temp/poem', 'r', encoding='utf8').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

hidden_size = 100
seq_length = 25
learning_rage = 0.1


def initialize():
    with tf.variable_scope('hidden'):
        Wxh = tf.get_variable('Wxh', (hidden_size, vocab_size),
                              initializer=tf.truncated_normal_initializer)
        Whh = tf.get_variable('Whh', (hidden_size, hidden_size),
                              initializer=tf.truncated_normal_initializer)
        bh = tf.get_variable('bh', (hidden_size, 1),
                             initializer=tf.zeros_initializer)
    with tf.variable_scope('out'):
        Why = tf.get_variable('Why', (vocab_size, hidden_size),
                              initializer=tf.truncated_normal_initializer)
        by = tf.get_variable('by', (vocab_size, 1),
                             initializer=tf.zeros_initializer)

    return (Wxh, Whh, bh, Why, by)

parameters = initialize()
inputs0 = tf.placeholder(tf.int32, (seq_length, vocab_size), 'inputs')
targets0 = tf.placeholder(tf.int32, (seq_length, vocab_size), 'targets')
h_prev0 = tf.placeholder(tf.float32, (hidden_size, 1), 'h_prev')


def cell(x, h_prev, parameters):
    (Wxh, Whh, bh, Why, by) = parameters
    x = tf.reshape(x, (vocab_size, 1))
    h_curr = tf.tanh(tf.matmul(Wxh, tf.cast(x, tf.float32)) + tf.matmul(Whh, tf.cast(h_prev, tf.float32)) + bh)
    y = tf.matmul(Why, h_curr) + by
    return y


def get_loss(y_hat, targets_one_hot):
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=targets_one_hot, logits=y_hat))


def rnn_model(parameters, inputs_one_hot, targets_one_hot, h_prev):
    """

    :param inputs: array, seq_length length
    :param targets: array, seq_length length
    :param h_prev: previous hidden output
    :return:
    """
    word_num = seq_length
    print(inputs_one_hot.get_shape())
    assert(inputs_one_hot.get_shape() == (word_num, vocab_size))
    assert(h_prev.shape == (hidden_size, 1))

    tf.reshape(inputs_one_hot, (word_num, vocab_size, 1))
    loss = tf.zeros((1))
    for t in range(word_num):
        with tf.variable_scope('hidden', reuse=True):
            y_hat = cell(inputs_one_hot[t], h_prev, parameters)

            target = tf.reshape(targets_one_hot[t], (vocab_size, 1))
            curr_loss = get_loss(y_hat, target)
            loss = tf.add(loss, curr_loss)
    return loss


def sample(sess, parameters, h_prev, inputs_one_host, n):
    x = inputs_one_host[0]
    output_list = []
    for t in range(n):
        y_hat = cell(x, h_prev, parameters)
        assert (y_hat.get_shape() == (vocab_size, 1))
        max_index = tf.argmax(y_hat)
        y_hat = tf.one_hot(max_index, vocab_size)
        assert(y_hat.get_shape() == (1, vocab_size))
        output_list.append(sess.run(max_index, feed_dict={h_prev0: h_prev, inputs0: inputs_one_host}))
        x = tf.squeeze(y_hat)

    return output_list


loss = rnn_model(parameters, inputs0, targets0, h_prev0)
train_op = tf.train.AdamOptimizer(learning_rage).minimize(loss)

# sample_op = sample(parameters, h_prev0, inputs0, 30)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    n, p = 0, 0
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length
    while True:
        if p + seq_length + 1 >= len(data) or n == 0:
            h_prev = np.zeros((hidden_size, 1))
            p = 0

        inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

        # convert to one hot style
        num_class = vocab_size
        inputs = np.array(inputs).reshape(-1)
        input_one_hots = np.eye(num_class)[inputs]

        targets = np.array(targets).reshape(-1)
        target_one_hots = np.eye(num_class)[targets]
        assert(input_one_hots.shape == (seq_length, num_class))
        assert (target_one_hots.shape == (seq_length, num_class))

        _, loss_train = sess.run([train_op, loss], feed_dict={inputs0: input_one_hots,
                                      targets0:target_one_hots,
                                      h_prev0:h_prev})

        smooth_loss = smooth_loss * 0.999 + loss_train * 0.001
        if n % 100 == 0: print('iter %d, loss: %s' % (n, smooth_loss))

        # sample from the model now and then
        if n % 100 == 0:
            sample_ix = sample(sess, parameters, h_prev, input_one_hots, 30)
            sample_ix = np.squeeze(sample_ix)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('----\n %s \n----' % (txt,))

        n += 1
        p += seq_length



































