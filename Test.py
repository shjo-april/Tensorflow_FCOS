# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import glob
import time
import random

import numpy as np
import tensorflow as tf

from Utils import *
from Define import *

from FCOS import *
from FCOS_Loss import *
from FCOS_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
# train_xml_paths = [ROOT_DIR + line.strip() for line in open('./dataset/train.txt', 'r').readlines()]
test_xml_paths = [ROOT_DIR + line.strip() for line in open('./dataset/valid.txt', 'r').readlines()]
test_xml_count = len(test_xml_paths)

log_print('[i] test : {}'.format(test_xml_count))

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

fcos_dic, fcos_sizes = FCOS(input_var, False)
fcos_utils = FCOS_Utils(fcos_sizes)

pred_bboxes_op = fcos_dic['pred_bboxes']
pred_classes_op = fcos_dic['pred_classes']

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/FCOS_115000.ckpt')

for test_iter, xml_path in enumerate(test_xml_paths):
    image_path, gt_bboxes, gt_classes = xml_read(xml_path, CLASS_NAMES)

    image = cv2.imread(image_path)
    h, w, c = image.shape

    tf_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
    decode_bboxes, decode_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : [tf_image]})

    pred_bboxes = fcos_utils.Decode(decode_bboxes[0], decode_classes[0], [w, h], detect_threshold = 0.20, use_nms = True)
    
    for pred_bbox in pred_bboxes:
        xmin, ymin, xmax, ymax = pred_bbox[:4].astype(np.int32)
        confidence = pred_bbox[4] * 100
        class_name = CLASS_NAMES[int(pred_bbox[5])]
        
        cv2.putText(image, '{} = {:.2f}%'.format(class_name, confidence), (xmin, ymin - 10), 1, 1, (0, 255, 0), 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    cv2.imshow('show', image)
    cv2.waitKey(0)

