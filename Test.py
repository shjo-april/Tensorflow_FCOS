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

from Define import *
from Utils import *
from Teacher import *

from FCOS import *
from FCOS_Loss import *
from FCOS_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
test_data_list = np.load('./dataset/train_detection.npy', allow_pickle = True)

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
fcos_dic, fcos_sizes = FCOS(input_var, False)
fcos_utils = FCOS_Utils(fcos_sizes)

pred_bboxes_op = fcos_dic['pred_bboxes']
pred_centers_op = fcos_dic['pred_centers']
pred_classes_op = fcos_dic['pred_classes']

# pred_bboxes_op = FCOS_Decode_Layer(pred_bboxes_op, fcos_utils.centers)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/FCOS_{}.ckpt'.format(105000))

for data in test_data_list:
    image_name, gt_bboxes, gt_classes = data

    gt_bboxes = np.asarray(gt_bboxes, dtype = np.int32)

    image = cv2.imread(TRAIN_DIR + image_name)
    h, w, c = image.shape

    tf_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
    total_pred_bboxes, total_pred_centers, total_pred_classes = sess.run([pred_bboxes_op, pred_centers_op, pred_classes_op], feed_dict = {input_var : [tf_image]})
    
    pred_bboxes, pred_classes = fcos_utils.Decode(total_pred_bboxes[0], total_pred_centers[0], total_pred_classes[0], [w, h], detect_threshold = 0.20)
    
    for bbox, class_index in zip(pred_bboxes, pred_classes):
        xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
        conf = bbox[4]
        class_name = CLASS_NAMES[class_index]

        string = "{} : {:.2f}%".format(class_name, conf * 100)
        cv2.putText(image, string, (xmin, ymin - 10), 1, 1, (0, 255, 0))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC)

    cv2.imshow('show', image)
    cv2.waitKey(0)
