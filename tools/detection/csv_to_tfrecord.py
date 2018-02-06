import csv
import os

import tensorflow as tf

INPUT_FILE = "output.csv"
TRAIN_RATIO = 0.8


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def create_tf_example(filename, data, xmin, xmax, ymin, ymax, img_size):
    filename = filename.encode("utf8")

    xmins = [xmin / img_size[0]]
    xmaxs = [xmax / img_size[0]]
    ymins = [ymin / img_size[1]]
    ymaxs = [ymax / img_size[1]]

    classes_text = ["subtitle".encode("utf8")]
    classes = [1]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": int64_feature(img_size[1]),
        "image/width": int64_feature(img_size[0]),
        "image/filename": bytes_feature(filename),
        "image/source_id": bytes_feature(filename),
        "image/encoded": bytes_feature(data),
        "image/format": bytes_feature(b"jpeg"),
        "image/object/bbox/xmin": float_list_feature(xmins),
        "image/object/bbox/xmax": float_list_feature(xmaxs),
        "image/object/bbox/ymin": float_list_feature(ymins),
        "image/object/bbox/ymax": float_list_feature(ymaxs),
        "image/object/class/text": bytes_list_feature(classes_text),
        "image/object/class/label": int64_list_feature(classes),
    }))

    return tf_example


def main():
    writer_train = tf.python_io.TFRecordWriter("train.record")
    writer_test = tf.python_io.TFRecordWriter("validation.record")

    lines = sum(1 for line in open(INPUT_FILE))

    with open(INPUT_FILE, "r") as f:
        reader = csv.reader(f, delimiter=',')
        i = 1
        for row in reader:
            print("{}/{}".format(i, lines), end="\r")
            filename, path, height, width, name, xmin, xmax, ymin, ymax = row
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            img_size = (int(width), int(height))
            if not os.path.isfile(path):
                print("{} not found".format(path))
                continue

            with open(path, "rb") as f:
                data = f.read()
                tf_example = create_tf_example(filename, data, xmin, xmax, ymin, ymax, img_size)

                if i < int(lines * TRAIN_RATIO):
                    writer_train.write(tf_example.SerializeToString())
                else:
                    writer_test.write(tf_example.SerializeToString())
            i += 1

    writer_train.close()
    writer_test.close()

    with open("label_map.pbtxt", "w") as label_map:
        label_map.write("item {\n")
        label_map.write("  id: 1\n")
        label_map.write("  name: 'subtitle'\n")
        label_map.write("}")


if __name__ == "__main__":
    main()
