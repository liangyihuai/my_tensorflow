import tensorflow as tf
from com.huai.converlution.resnets.resnets_utils import load_dataset
from com.huai.converlution.resnets.resnets_utils import process_orig_datasets
import numpy as np




def identity_block(X, kernel_size, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("id_block_stage"+str(stage)):
        filter1, filter2, filter3 = filters
        X_shortcut = X

        # First component of main path
        X = tf.layers.conv2d(X, filter1, kernel_size=(1, 1), strides=(1, 1),name=conv_name_base+'2a',
                         kernel_initializer=tf.glorot_uniform_initializer(seed=0))
        X = tf.layers.batch_normalization(X, axis=3, name=bn_name_base+'2a')
        X = tf.nn.relu(X)

        # Second component of main path
        X = tf.layers.conv2d(X, filter2, (kernel_size, kernel_size), padding='same',
                             name=conv_name_base+'2b',
                             kernel_initializer=tf.glorot_uniform_initializer(seed=0))
        X = tf.layers.batch_normalization(X, axis=3, name=bn_name_base+'2b')
        X = tf.nn.relu(X)

        # Third component of main path
        X = tf.layers.conv2d(X, filter3, kernel_size=(1, 1),
                             name=conv_name_base+'2c', kernel_initializer=tf.glorot_uniform_initializer(seed=0))
        X = tf.layers.batch_normalization(X, axis=3, name=bn_name_base + '2c')

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = tf.add(X, X_shortcut)
        X = tf.nn.relu(X)

    return X


def convolutional_block(X, kernel_size, filters, stage, block, stride = 2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    stride -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):

        # Retrieve Filters
        filter1, filter2, filter3 = filters

        # Save the input value
        X_shortcut = X

        # First component of main path
        X = tf.layers.conv2d(X, filter1, kernel_size=(1, 1), strides=(stride, stride), name=conv_name_base+'2a',
                             kernel_initializer=tf.glorot_uniform_initializer(seed=(0)))
        X = tf.layers.batch_normalization(X, axis=3, name=bn_name_base+'2a')
        X = tf.nn.relu(X)

        # Second component of main path
        X = tf.layers.conv2d(X, filter2, (kernel_size, kernel_size), name=conv_name_base + '2b',
                             padding='same', kernel_initializer=tf.glorot_uniform_initializer(seed=0))
        X = tf.layers.batch_normalization(X, axis=3, name=bn_name_base + '2b')
        X = tf.nn.relu(X)

        # Third component of main path
        X = tf.layers.conv2d(X, filter3, (1, 1),
                             name=conv_name_base + '2c', kernel_initializer=tf.glorot_uniform_initializer(seed=0))
        X = tf.layers.batch_normalization(X, axis=3, name=bn_name_base + '2c')

        # SHORTCUT PATH
        X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1,1), strides=(stride, stride), name=conv_name_base + '1',
                                      kernel_initializer=tf.glorot_uniform_initializer(seed=0))
        X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1')

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = tf.add(X_shortcut, X)
        X = tf.nn.relu(X)

    return X


def ResNet50(X_train, Y_train, X_test, Y_test, classes= 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:

    Returns:
    """
    m, H_size, W_size, C_size = X_train.shape
    X = tf.placeholder(tf.float32, shape=(None, H_size, W_size, C_size), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, classes), name='Y')

    X = tf.pad(X, tf.constant([[0, 0],[3, 3,], [3, 3], [0, 0]]), "CONSTANT")

    # stage 1
    X = tf.layers.conv2d(X, filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', kernel_initializer=tf.glorot_uniform_initializer(seed=0))
    X = tf.layers.batch_normalization(X, axis=3, name='bn_conv1')
    X = tf.nn.relu(X)
    X = tf.layers.max_pooling2d(X, pool_size=(3, 3), strides=(2, 2))

    # stage 2
    X = convolutional_block(X, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', stride=1)
    X = identity_block(X, 3, [4, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [4, 64, 256], stage=2, block='c')

    # stage 3
    X = convolutional_block(X, kernel_size=3, filters=[128,128,512], stage=3, block='a', stride=2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')

    # stage 4
    X = convolutional_block(X, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', stride=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # stage 5
    X = convolutional_block(X, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a', stride=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = tf.layers.average_pooling2d(X, pool_size=(2, 2), strides=(1,1))

    X = tf.layers.flatten(X, name='flatten')
    X = tf.layers.dense(X, classes, activation=tf.nn.softmax, name='fc' + str(classes), kernel_initializer=tf.glorot_uniform_initializer(seed=0))

    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=X))
    train_op = tf.train.AdamOptimizer(0.01).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10):
            _, cost_sess = sess.run([train_op, cost], feed_dict={X:X_train, Y:Y_train})
            print(cost_sess)


path = 'D:/LiangYiHuai/deepleanring/resnets/datasets'
orig_data = load_dataset(path)

X_train, Y_train, X_test, Y_test = process_orig_datasets(orig_data)
"""
number of training examples = 1080
number of test examples = 120
X_train shape: (1080, 64, 64, 3)
Y_train shape: (1080, 6)
X_test shape: (120, 64, 64, 3)
Y_test shape: (120, 6)"""

ResNet50(X_train, Y_train, X_test, Y_test)

