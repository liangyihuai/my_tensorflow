import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

log_dir = "C:\\Users\\liangyh\\Desktop\\temp\\"

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Input placeholders
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
Y_train = tf.placeholder(tf.float32, [None, 10], name='y-input')
is_training = tf.placeholder(tf.bool)

# inut layer
input_layer = tf.reshape(x, [-1, 28, 28, 1])
with tf.name_scope('conv1'):
    #Convlution #1 ([5,5] : [28x28x1]->[28x28x6])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=6,
        kernel_size=[5, 5],
        padding="same",
        activation=None
    )

    #Batch Norm #1
    conv1_bn = tf.layers.batch_normalization(
        inputs=conv1,
        axis=-1,
        momentum=0.9,
        epsilon=0.001,
        center=True,
        scale=True,
        training = is_training,
        name='conv1_bn'
    )

    #apply relu
    conv1_bn_relu = tf.nn.relu(conv1_bn)
    #apply pool ([2,2] : [28x28x6]->[14X14X6])
    maxpool1=tf.layers.max_pooling2d(
        inputs=conv1_bn_relu,
        pool_size=[2,2],
        strides=2,
        padding="valid"
    )

with tf.name_scope('conv2'):
    #convolution #2 ([5x5] : [14x14x6]->[14x14x16]
    conv2 = tf.layers.conv2d(
        inputs=maxpool1,
        filters=16,
        kernel_size=[5, 5],
        padding="same",
        activation=None
    )

    #Batch Norm #2
    conv2_bn = tf.layers.batch_normalization(
        inputs=conv2,
        axis=-1,
        momentum=0.999,
        epsilon=0.001,
        center=True,
        scale=True,
        training = is_training
    )

    #apply relu
    conv2_bn_relu = tf.nn.relu(conv2_bn)
    #maxpool2 ([2,2] : [14x14x16]->[7x7x16]
    maxpool2=tf.layers.max_pooling2d(
        inputs=conv2_bn_relu,
        pool_size=[2,2],
        strides=2,
        padding="valid"
    )

#fully connected 1 [7*7*16 = 784 -> 120]
maxpool2_flat=tf.reshape(maxpool2,[-1,7*7*16])
fc1 = tf.layers.dense(
    inputs=maxpool2_flat,
    units=120,
    activation=None
)

#Batch Norm #2
fc1_bn = tf.layers.batch_normalization(
    inputs=fc1,
    axis=-1,
    momentum=0.999,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training
)
#apply reliu

fc1_bn_relu = tf.nn.relu(fc1_bn)

#fully connected 2 [120-> 84]
fc2 = tf.layers.dense(
    inputs=fc1_bn_relu,
    units=84,
    activation=None
)

#apply relu
fc2_bn_relu = tf.nn.relu(fc2)

#fully connected 3 [84->10]. Output layer with softmax
logits = tf.layers.dense(
    inputs=fc2_bn_relu,
    units=10,
    activation=None)

#loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_train, logits=logits))
tf.summary.scalar('cross entropy', cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_train, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy',accuracy)

#merge summaries and init train writer
sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train' ,sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess.run(init)

with sess.as_default():
    def get_variables_values():
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        values = {}
        for variable in variables:
            values[variable.name[:-2]] = sess.run(variable, feed_dict={
                x:batch[0], Y_train:batch[1], is_training:True
                })
        return values


    for i in range(1000):
        batch = mnist.train.next_batch(64)
        if i%100 == 0: #test-set summary
            print('####################################')
            values = get_variables_values()
            print('moving variance is:')
            print(values["conv1_bn/moving_variance"])
            print('moving mean is:')
            print(values["conv1_bn/moving_mean"])
            print('gamma is:')
            print(values["conv1_bn/gamma/Adam"])
            print('beta is:')
            print(values["conv1_bn/beta/Adam"])
            summary, acc = sess.run([merged,accuracy], feed_dict={
                x:mnist.test.images, Y_train:mnist.test.labels, is_training:False})

        else:
            summary, _ = sess.run([merged,train_step], feed_dict={x:batch[0], Y_train:batch[1], is_training:True})
            if i%10 == 0:
                train_writer.add_summary(summary,i)