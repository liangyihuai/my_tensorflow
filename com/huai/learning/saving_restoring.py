import tensorflow as tf;
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util;


SAVE_PATH = 'C:\\Users\\liangyh\\Desktop\\temp\\temp_model\\model.ckpt'
MOVING_AVERAGE_SAVE_PATH = "C:\\Users\\liangyh\\Desktop\\temp\\temp_model2\\model.ckpt";
MODEL_SAVE_PATH = "C:\\Users\\liangyh\\Desktop\\temp\\temp_model3\\model.ckpt";

def saving_variables():
    v1 = tf.get_variable('v1', shape=[3], initializer=tf.random_normal_initializer);
    v2 = tf.get_variable('v2', shape=[5], initializer=tf.random_normal_initializer);

    inc_v1 = v1.assign(v1+1);
    doc_v2 = v2.assign(v2-1);

    init_op = tf.global_variables_initializer();

    saver = tf.train.Saver();

    with tf.Session() as sess:
        sess.run(init_op);
        # sess.run(v1);
        # sess.run(v2);
        save_path = saver.save(sess, save_path=SAVE_PATH);
        print("model saved in file: %s"%(save_path));


def restoring_variables():
    tf.reset_default_graph();

    v1 = tf.get_variable('v1', shape=[3]);
    v2 = tf.get_variable('v2', shape=[5]);

    saver = tf.train.Saver();

    with tf.Session() as sess:
        saver.restore(sess, SAVE_PATH);
        print("model restored");
        print("v1: %s" % v1.eval());
        print("v2: %s" % v2.eval());


def choosing_variable_to_save():
    v1 = tf.get_variable('v1', shape=[3], initializer=tf.random_normal_initializer);
    v2 = tf.get_variable('v2', shape=[5], initializer=tf.random_normal_initializer);

    inc_v1 = v1.assign(v1 + 1);
    doc_v2 = v2.assign(v2 - 1);

    saver = tf.train.Saver({'v2': v2});

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer();
        sess.run(init_op);
        # sess.run(v1);
        # sess.run(v2);
        save_path = saver.save(sess, save_path=SAVE_PATH);
        print("model saved in file: %s" % (save_path));


def choosing_variable_to_restore():
    """You can create as many Saver objects as you want if you need to save and restore different subsets
    of the model variables. The same variable can be listed in multiple saver objects; its value is only
    changed when the Saver.restore() method is run.

    If you only restore a subset of the model variables at the start of a session, you have to run an initialize
    op for the other variables. See tf.variables_initializer for more information.

    To inspect the variables in a checkpoint, you can use the inspect_checkpoint library, particularly the
    print_tensors_in_checkpoint_file function.

    By default, Saver uses the value of the tf.Variable.name property for each variable. However, when you
    create a Saver object, you may optionally choose names for the variables in the checkpoint files."""
    tf.reset_default_graph();
    v1 = tf.get_variable('v1', [3], initializer=tf.zeros_initializer)
    v2 = tf.get_variable('v2', [5], initializer=tf.zeros_initializer);

    saver = tf.train.Saver({'v2': v2});

    with tf.Session() as sess:
        # v1.initializer.run()
        sess.run(tf.global_variables_initializer());
        saver.restore(sess, SAVE_PATH);
        print("v1: %s" % v1.eval());
        print("v2: %s" % v2.eval());


def change_name_while_restore():
    tf.reset_default_graph();
    v1 = tf.get_variable('v1', [3], initializer=tf.zeros_initializer)
    other_v2 = tf.get_variable('other-v2', [5], initializer=tf.zeros_initializer);

    saver = tf.train.Saver({'v2': other_v2});

    with tf.Session() as sess:
        # v1.initializer.run()
        sess.run(tf.global_variables_initializer());
        saver.restore(sess, SAVE_PATH);
        print("v1: %s" % v1.eval());
        print("v2: %s" % other_v2.eval());


def save_moving_average():
    tf.reset_default_graph();
    v = tf.Variable(0, dtype=tf.float32, name='v');
    for variables in tf.all_variables():
        print(variables.name);
    ema = tf.train.ExponentialMovingAverage(0.99);
    maintain_averages_op = ema.apply(tf.all_variables());
    for variables in tf.all_variables():
        print(variables.name)
    saver = tf.train.Saver();
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());

        sess.run(tf.assign(v, 10));
        sess.run(maintain_averages_op);
        saver.save(sess, MOVING_AVERAGE_SAVE_PATH);
        print(sess.run([v, ema.average(v)]));


def restore_moving_average0():
    v = tf.Variable(0, dtype=tf.float32, name='v');
    saver = tf.train.Saver({'v/ExponentialMovingAverage': v});
    with tf.Session() as sess:
        saver.restore(sess, MOVING_AVERAGE_SAVE_PATH);
        print(sess.run(v));


def restore_moving_average():
    v = tf.Variable(0, dtype=tf.float32, name='v');
    ema = tf.train.ExponentialMovingAverage(0.99);
    print(ema.variables_to_restore());
    # can not change the variable name.
    saver = tf.train.Saver(ema.variables_to_restore());
    with tf.Session() as sess:
        saver.restore(sess, MOVING_AVERAGE_SAVE_PATH);
        print(sess.run(v));


def save_graph():
    v1 = tf.Variable(1, dtype=tf.float32, name='v1');
    v2 = tf.Variable(2, dtype=tf.float32, name='v2');
    # note: result is a node of the graph, not a variable.
    result = v1 + v2;

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        graph_def = tf.get_default_graph().as_graph_def();

        output_graph_def = graph_util.convert_variables_to_constants(
            sess, graph_def, ['add'])
        with tf.gfile.GFile(MODEL_SAVE_PATH, 'wb') as f:
            f.write(output_graph_def.SerializeToString())


def restore_graph():
    with tf.Session() as sess:
        with gfile.FastGFile(MODEL_SAVE_PATH, 'rb') as f:
            graph_def = tf.GraphDef();
            graph_def.ParseFromString(f.read());
            result = tf.import_graph_def(graph_def, return_elements=['add:0'])
            print(sess.run(result))


def save_model():
    builder = tf.saved_model_builder.SavedModelBuilder();



# restoring_variables();
# choosing_variable_to_save()
# choosing_variable_to_restore();
# change_name_while_restore();
# save_moving_average();
# restore_moving_average()

save_graph();
restore_graph();










