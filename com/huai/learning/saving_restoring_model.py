import tensorflow as tf
import os ;

model_saving_path = "C:\\Users\\liangyh\\Desktop\\temp\\temp_model4\\"
model_name = 'saving_restoring';


def save():
    w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
    w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
    saver = tf.train.Saver([w1,w2])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, os.path.join(model_saving_path, model_name), global_step=1000)


def restoring():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(model_saving_path, model_name+'-1000.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(model_saving_path))
        print(sess.run('w1:0'))
        # Model has been restored. Above statement will print the saved value of w1.

save();
restoring();