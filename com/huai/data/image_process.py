# coding=utf-8

import matplotlib.pyplot as plt;
import tensorflow as tf;
import numpy as np;

image_file = 'C:/Users/USER/Downloads/sample.jpg';
image_file_output = 'C:/Users/USER/Downloads/sample2.jpg';
image_file_png = 'C:/Users/USER/Downloads/sample.png';

def read_write_raw_image():
    image_raw_data = tf.gfile.FastGFile(image_file, 'rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())

    plt.figure(1)
    plt.imshow(img_data.eval())
    plt.show()

    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)

    encoded_image = tf.image.encode_jpeg(img_data);
    with tf.gfile.GFile(image_file_output, 'wb') as f:
        f.write(encoded_image.eval())


def resize_images():
    image_raw_data = tf.gfile.FastGFile(image_file, 'rb').read()

    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        resized = tf.image.resize_images(img_data, [500, 500], method=0)
        resized = np.asarray(resized.eval(), dtype='uint8');
        plt.figure(2)
        plt.imshow(resized);
        plt.show()

        # 如果目标图像小于原始图像，那么就会在原始图像的基础之上填充0
        croped = tf.image.resize_image_with_crop_or_pad(img_data, 300, 300);
        padded = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 2000);
        plt.figure(3)
        plt.imshow(croped.eval());
        plt.show()
        plt.figure(4)
        plt.imshow(padded.eval());
        plt.show();

        central_cropped = tf.image.central_crop(img_data, 0.6)
        plt.figure(5)
        plt.imshow(central_cropped.eval())
        plt.show();


def flip():
    image_raw_data = tf.gfile.FastGFile(image_file, 'rb').read()

    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        # flipped = tf.image.flip_up_down(img_data)
        flipped = tf.image.random_flip_up_down(img_data)
        plt.imshow(flipped.eval())
        plt.show();

        # left_right_flip = tf.image.flip_left_right(img_data);
        left_right_flip = tf.image.random_flip_left_right(img_data);
        plt.imshow(left_right_flip.eval());
        plt.show()

        transposed = tf.image.transpose_image(img_data);
        plt.imshow(transposed.eval());
        plt.show()


def bounding_box():
    image_raw_data = tf.gfile.FastGFile(image_file, 'rb').read()

    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        # tf.image.resize_images(img_data, [500, 500], method=0)
        batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0);
        boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

        result = tf.image.draw_bounding_boxes(batched, boxes);

        plt.imshow(result[0].eval());
        plt.show()


def random_bounding_box():
    image_raw_data = tf.gfile.FastGFile(image_file, 'rb').read()

    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
            tf.shape(img_data), bounding_boxes=boxes);
        batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0);
        image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
        for i in range(10):

            distorted_image = tf.slice(img_data, begin, size)

            plt.imshow(distorted_image.eval());
            plt.show()



# resize_images()
# flip();
# bounding_box();
random_bounding_box()













