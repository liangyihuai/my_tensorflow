import tensorflow as tf;
from tensorflow.examples.tutorials.mnist import input_data;
import numpy as np;


def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

TFRecords_file = '../MNIST_data/output/output.tfrecords';

def write_TFRecords():
    # prepare the data to record
    mnist = input_data.read_data_sets('../MNIST_data', dtype=tf.uint8, one_hot=True);
    images = mnist.train.images;
    labels = mnist.train.labels;
    pixels = images.shape[1];
    num_examples = mnist.train.num_examples;

    file_name = TFRecords_file;
    # record tf data
    writer = tf.python_io.TFRecordWriter(file_name);
    for index in range(num_examples):
        image_raw = images[index].tostring();
        example = tf.train.Example(features=tf.train.Features(feature={
            'pixels':_int64_feature(pixels),
            'label':_int64_feature(np.argmax(labels[index])),
            'image_raw':_bytes_feature(image_raw)}))

        writer.write(example.SerializeToString())
    writer.close();


def read_TFRecords():
    reader = tf.TFRecordReader();
    file_name_queue = tf.train.string_input_producer([TFRecords_file]);
    _, serialized_example = reader.read(file_name_queue);
    features = tf.parse_single_example(serialized_example, features={
        'image_raw':tf.FixedLenFeature([], tf.string),
        'pixels':tf.FixedLenFeature([], tf.int64),
        'label':tf.FixedLenFeature([], tf.int64)})

    images = tf.decode_raw(features['image_raw'], tf.uint8);
    labels = tf.cast(features['label'], tf.int32);
    pixels = tf.cast(features['pixels'], tf.int32);

    sess = tf.Session();
    coord = tf.train.Coordinator();
    threads = tf.train.start_queue_runners(sess=sess, coord=coord);

    for i in range(10):
        image, label, pixel = sess.run([images, labels, pixels])
        print('----------------------')
        print(image)
        print(label)
        print(pixel)
        print('----------------------')



read_TFRecords();


















