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
import matplotlib.pyplot as plt

from Define import *
from Utils import *

from FCOS import *
from FCOS_Utils import *

from mAP_Calculator import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
test_xml_paths = glob.glob(ROOT_DIR + 'VOC2007/train/xml/*.xml')[:1000]
test_xml_count = len(test_xml_paths)
print('[i] Test : {}'.format(len(test_xml_paths)))

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

fcos_dic, fcos_sizes = FCOS(input_var, False)
fcos_utils = FCOS_Utils(fcos_sizes)

pred_bboxes_op = fcos_dic['pred_bboxes']
pred_classes_op = fcos_dic['pred_classes']

# 3. create Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 4. restore Model
saver = tf.train.Saver()
saver.restore(sess, './model/FCOS_115000.ckpt')

# 5. calculate AP@50
mAP_calc = mAP_Calculator(classes = CLASSES)

test_time = time.time()

batch_image_data = []
batch_image_wh = []

batch_gt_bboxes = []
batch_gt_classes = []

for test_iter, xml_path in enumerate(test_xml_paths):
    image_path, gt_bboxes, gt_classes = xml_read(xml_path, CLASS_NAMES)

    ori_image = cv2.imread(image_path)
    image = cv2.resize(ori_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
    
    batch_image_data.append(image.astype(np.float32))
    batch_image_wh.append(ori_image.shape[:-1][::-1])

    batch_gt_bboxes.append(gt_bboxes)
    batch_gt_classes.append(gt_classes)

    # calculate correct/confidence
    if len(batch_image_data) == BATCH_SIZE:
        decode_bboxes, decode_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : batch_image_data})

        for i in range(BATCH_SIZE):
            gt_bboxes, gt_classes = batch_gt_bboxes[i], batch_gt_classes[i]
            pred_bboxes, pred_classes = fcos_utils.Decode(decode_bboxes[i], decode_classes[i], batch_image_wh[i], detect_threshold = 0.01, use_nms = True)

            if pred_bboxes.shape[0] == 0:
                pred_bboxes = np.zeros((0, 5), dtype = np.float32)

            mAP_calc.update(pred_bboxes, pred_classes, gt_bboxes, gt_classes)
        
        batch_image_data = []
        batch_image_wh = []

        batch_gt_bboxes = []
        batch_gt_classes = []
    
    sys.stdout.write('\r# Test = {:.2f}%'.format(test_iter / test_xml_count * 100))
    sys.stdout.flush()

if len(batch_image_data) != 0:
    decode_bboxes, decode_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : batch_image_data})

    for i in range(BATCH_SIZE):
        gt_bboxes, gt_classes = batch_gt_bboxes[i], batch_gt_classes[i]
        pred_bboxes, pred_classes = fcos_utils.Decode(decode_bboxes[i], decode_classes[i], batch_image_wh[i], detect_threshold = 0.01, use_nms = True)
        
        if pred_bboxes.shape[0] == 0:
            pred_bboxes = np.zeros((0, 5), dtype = np.float32)
        
        mAP_calc.update(pred_bboxes, pred_classes, gt_bboxes, gt_classes)

test_time = int(time.time() - test_time)
print('\n[i] test time = {}sec'.format(test_time))

map_list = []

for i, class_name in enumerate(CLASS_NAMES):
    ap, precisions, recalls, interp_list, precision_interp_list = mAP_calc.compute_precision_recall(i)

    # matplotlib (precision&recall curve + interpolation)
    plt.clf()
    
    plt.fill_between(recalls, precisions, step = 'post', alpha = 0.2, color = 'green')
    plt.plot(recalls, precisions, 'green')
    plt.plot(interp_list, precision_interp_list, 'ro')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('# Precision-recall curve ({} - {:.2f}%)'.format(class_name, ap))
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.show()
    plt.savefig('./results/{}.jpg'.format(class_name))

    map_list.append(ap)

'''
# AP@50 aeroplane = 62.81%
# AP@50 bicycle = 61.96%
# AP@50 bird = 57.04%
# AP@50 boat = 42.30%
# AP@50 bottle = 33.26%
# AP@50 bus = 65.36%
# AP@50 car = 63.38%
# AP@50 cat = 76.04%
# AP@50 chair = 41.93%
# AP@50 cow = 53.36%
# AP@50 diningtable = 50.93%
# AP@50 dog = 71.16%
# AP@50 horse = 62.82%
# AP@50 motorbike = 66.85%
# AP@50 person = 59.76%
# AP@50 pottedplant = 31.92%
# AP@50 sheep = 56.05%
# AP@50 sofa = 54.13%
# AP@50 train = 74.72%
# AP@50 tvmonitor = 62.69%
# mAP@50 = 57.42%
'''
print()
for ap, class_name in zip(map_list, CLASS_NAMES):
    print('# AP@50 {} = {:.2f}%'.format(class_name, ap))
print('# mAP@50 = {:.2f}%'.format(np.mean(map_list)))
