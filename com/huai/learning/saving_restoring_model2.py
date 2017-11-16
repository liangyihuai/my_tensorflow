# -*- coding: UTF-8 -*-
import tensorflow as tf;
import os;

model_saving_path = "C:\\Users\\USER\\Desktop\\temp\\temp_model4"
model_name = 'saving_restoring';


def save():
    w1 = tf.placeholder(dtype=tf.float32, name='w1');
    w2 = tf.placeholder(dtype=tf.float32, name='w2');
    b1 = tf.Variable(2.0, name='bias');
    feed_dict = {w1:4, w2:8};

    w3 = tf.add(w1, w2)
    w4 = tf.multiply(w3, b1, name='op_to_restore');
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver();
        print(sess.run(w4, feed_dict));
        saver.save(sess, os.path.join(model_saving_path, model_name), global_step=1000);


def restore0():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            os.path.join(model_saving_path, model_name+'-1000.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(model_saving_path))

        graph = tf.get_default_graph();
        w1 = graph.get_tensor_by_name('w1:0');
        w2 = graph.get_tensor_by_name('w2:0');
        feed_dict = {w1:13.0, w2:17.0};

        op_to_restore = graph.get_tensor_by_name('op_to_restore:0');
        print(sess.run(op_to_restore, feed_dict))


def restore2():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            os.path.join(model_saving_path, model_name+'-1000.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(model_saving_path))

        graph = tf.get_default_graph();
        w1 = graph.get_tensor_by_name('w1:0');
        w2 = graph.get_tensor_by_name('w2:0');
        feed_dict = {w1:13.0, w2:17.0};

        op_to_restore = graph.get_tensor_by_name('op_to_restore:0');
        # Add more to the current graph
        add_on_op = tf.multiply(op_to_restore, 2)
        print(sess.run(add_on_op, feed_dict))
        # This will print 120.

# save()
restore2();








