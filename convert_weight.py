# coding: utf-8
# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np

from model import yolov3
from utils.misc_utils import parse_anchors, load_weights

args = sys.argv[1:]

#num_class = 80
#weight_path = './data/darknet_weights/yolov3.weights'
#save_path = './data/darknet_weights/yolov3.ckpt'
weight_path = args[0]
save_path = args[1]
num_class = int(args[2])
img_size = int(args[3])
anchors = args[4] if len(args) > 4 else './data/yolo_anchors.txt'
anchors = parse_anchors(anchors)

model = yolov3(num_class, anchors)
from utils.tflex import Session

with Session() as sess:
    inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

    with tf.variable_scope('yolov3'):
        feature_map = model.forward(inputs)

    saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

    load_ops = load_weights(tf.global_variables(scope='yolov3'), weight_path)
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))



