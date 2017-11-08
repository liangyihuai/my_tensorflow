import tensorflow as tf


def save_model():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1");
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2");
    result = v1 + v2;

    init_op = tf.global_variables_initializer();
    saver = tf.train.Saver();

    with tf.Session() as sess:
        sess.run(init_op)
        saver.save(sess, "C:\\Users\\liangyh\\Desktop\\temp\\model.ckpt")


def restore_model():
    saver = tf.train.import_meta_graph(
        "C:\\Users\\liangyh\\Desktop\\temp\\model.ckpt.meta");
    with tf.Session() as sess:
        saver.restore(sess, "C:\\Users\\liangyh\\Desktop\\temp\\model.ckpt")
        print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))


restore_model();
















