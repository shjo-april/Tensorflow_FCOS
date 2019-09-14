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
# from Teacher import *
from DataAugmentation import *

from DSSD import *
from DSSD_Loss import *
from DSSD_Utils import *

NUM_GPU = 2
BATCH_SIZE *= NUM_GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 1. dataset
train_xml_paths = [ROOT_DIR + line.strip() for line in open('./dataset/train.txt', 'r').readlines()]
valid_xml_paths = [ROOT_DIR + line.strip() for line in open('./dataset/valid.txt', 'r').readlines()]
valid_xml_count = len(valid_xml_paths)

open('log.txt', 'w')
log_print('[i] Train : {}'.format(len(train_xml_paths)))
log_print('[i] Valid : {}'.format(len(valid_xml_paths)))

# 2. build
input_var = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
is_training = tf.placeholder(tf.bool)

input_vars = tf.split(input_var, NUM_GPU)

pred_bboxes_ops = []
pred_classes_ops = []

for gpu_id in range(NUM_GPU):
    reuse = gpu_id != 0
    
    with tf.device(tf.DeviceSpec(device_type = "GPU", device_index = gpu_id)):
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            print(input_vars[gpu_id], is_training, reuse)
            dssd_dic, dssd_sizes = DSSD(input_vars[gpu_id], is_training, reuse)

            if not reuse:
                anchors = generate_anchors(dssd_sizes, [IMAGE_WIDTH, IMAGE_HEIGHT], ANCHOR_SCALES, ANCHOR_RATIOS)

            pred_bboxes_op = Decode_Layer(dssd_dic['pred_bboxes'], anchors)
            pred_classes_op = dssd_dic['pred_classes']
            
            pred_bboxes_ops.append(pred_bboxes_op)
            pred_classes_ops.append(pred_classes_op)

pred_bboxes_op = tf.concat(pred_bboxes_ops, axis = 0)
pred_classes_op = tf.concat(pred_classes_ops, axis = 0)

gt_bboxes_var = tf.placeholder(tf.float32, [BATCH_SIZE, anchors.shape[0], 4])
gt_classes_var = tf.placeholder(tf.float32, [BATCH_SIZE, anchors.shape[0], CLASSES])

log_print('[i] pred_bboxes_op : {}'.format(pred_bboxes_op))
log_print('[i] pred_classes_op : {}'.format(pred_classes_op))
log_print('[i] gt_bboxes_var : {}'.format(gt_bboxes_var))
log_print('[i] gt_classes_var : {}'.format(gt_classes_var))

loss_op, focal_loss_op, giou_loss_op = SSD_Loss(pred_bboxes_op, pred_classes_op, gt_bboxes_var, gt_classes_var)

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY
loss_op += l2_reg_loss_op

tf.summary.scalar('loss', loss_op)
tf.summary.scalar('focal_loss', focal_loss_op)
tf.summary.scalar('giou_loss', giou_loss_op)
tf.summary.scalar('l2_regularization_Loss', l2_reg_loss_op)
summary_op = tf.summary.merge_all()

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    # train_op = tf.train.MomentumOptimizer(learning_rate_var, momentum = 0.9).minimize(loss_op, colocate_gradients_with_ops = True)
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op, colocate_gradients_with_ops = True)

# 3. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# '''
pretrained_vars = []
for var in vars:
    if 'resnet_v2_50' in var.name:
        pretrained_vars.append(var)

pretrained_saver = tf.train.Saver(var_list = pretrained_vars)
pretrained_saver.restore(sess, './resnet_v2_model/resnet_v2_50.ckpt')
# '''

saver = tf.train.Saver()
# saver.restore(sess, './model/SSD_ResNet_v2_{}.ckpt'.format(30000))

best_valid_mAP = 0.0
learning_rate = INIT_LEARNING_RATE

train_iteration = len(train_xml_paths) // BATCH_SIZE
valid_iteration = len(valid_xml_paths) // BATCH_SIZE

max_iteration = train_iteration * MAX_EPOCH
decay_iteration = np.asarray([0.5 * max_iteration, 0.75 * max_iteration], dtype = np.int32)

log_print('[i] max_iteration : {}'.format(max_iteration))
log_print('[i] decay_iteration : {}'.format(decay_iteration))

loss_list = []
focal_loss_list = []
giou_loss_list = []
l2_reg_loss_list = []
train_time = time.time()

train_writer = tf.summary.FileWriter('./logs/train')

# train_threads = []
# for i in range(5):
#     train_thread = Teacher(train_xml_paths, anchors, max_data_size = 20, debug = True)
#     train_thread.start()
#     train_threads.append(train_thread)
# input()

for iter in range(1, max_iteration + 1):
    if iter in decay_iteration:
        learning_rate /= 10
        log_print('[i] learning rate decay : {} -> {}'.format(learning_rate * 10, learning_rate))

    ## Teacher (Thread)
    # find = False
    # while not find:
    #     for train_thread in train_threads:
    #         if train_thread.ready:
    #             find = True
    #             batch_image_data, batch_gt_bboxes, batch_gt_classes = train_thread.get_batch_data()        
    #             break

    ## Default
    batch_image_data = []
    batch_gt_bboxes = []
    batch_gt_classes = []
    batch_xml_paths = random.sample(train_xml_paths, BATCH_SIZE)
    
    for xml_path in batch_xml_paths:
        # delay = time.time()

        image, gt_bboxes, gt_classes = get_data(xml_path, training = True, augment = True)
        
        # image = image.astype(np.uint8)
        # for bbox in gt_bboxes:
        #     xmin, ymin, xmax, ymax = bbox
        #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # cv2.imshow('show', image)
        # cv2.waitKey(0)
        
        # delay = time.time() - delay
        # print('[D] {} = {}ms'.format('xml', int(delay * 1000))) # ~ 41ms

        encode_bboxes, encode_classes = Encode(gt_bboxes, gt_classes, anchors)

        batch_image_data.append(image.astype(np.float32))
        batch_gt_bboxes.append(encode_bboxes)
        batch_gt_classes.append(encode_classes)

    batch_image_data = np.asarray(batch_image_data, dtype = np.float32) 
    batch_gt_bboxes = np.asarray(batch_gt_bboxes, dtype = np.float32)
    batch_gt_classes = np.asarray(batch_gt_classes, dtype = np.float32)

    _feed_dict = {input_var : batch_image_data, gt_bboxes_var : batch_gt_bboxes, gt_classes_var : batch_gt_classes, is_training : True, learning_rate_var : learning_rate}
    log = sess.run([train_op, loss_op, focal_loss_op, giou_loss_op, l2_reg_loss_op, summary_op], feed_dict = _feed_dict)
    # print(log[1:-1])
    
    if np.isnan(log[1]):
        print('[!]', log[1:-1])
        input()

    loss_list.append(log[1])
    focal_loss_list.append(log[2])
    giou_loss_list.append(log[3])
    l2_reg_loss_list.append(log[4])
    train_writer.add_summary(log[5], iter)

    if iter % LOG_ITERATION == 0:
        loss = np.mean(loss_list)
        focal_loss = np.mean(focal_loss_list)
        giou_loss = np.mean(giou_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] iter : {}, loss : {:.4f}, focal_loss : {:.4f}, giou_loss : {:.4f}, l2_reg_loss : {:.4f}, train_time : {}sec'.format(iter, loss, focal_loss, giou_loss, l2_reg_loss, train_time))

        loss_list = []
        focal_loss_list = []
        giou_loss_list = []
        l2_reg_loss_list = []
        train_time = time.time()

    if iter % VALID_ITERATION == 0:
        ap_threshold = 0.5
        nms_threshold = 0.6

        correct_dic = {}
        confidence_dic = {}
        all_ground_truths_dic = {}

        for class_name in CLASS_NAMES:
            correct_dic[class_name] = []
            confidence_dic[class_name] = []
            all_ground_truths_dic[class_name] = 0.

        batch_image_data = []
        batch_image_wh = []
        batch_gt_bboxes_dic = []

        valid_time = time.time()

        for valid_iter, xml_path in enumerate(valid_xml_paths):
            image_path, gt_bboxes_dic = class_xml_read(xml_path, CLASS_NAMES)

            ori_image = cv2.imread(image_path)
            image = cv2.resize(ori_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
            
            batch_image_data.append(image.astype(np.float32))
            batch_image_wh.append(ori_image.shape[:-1][::-1])
            batch_gt_bboxes_dic.append(gt_bboxes_dic)

            # calculate correct/confidence
            if len(batch_image_data) == BATCH_SIZE:
                encode_bboxes, encode_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : batch_image_data})

                for i in range(BATCH_SIZE):
                    gt_bboxes_dic = batch_gt_bboxes_dic[i]
                    for class_name in list(gt_bboxes_dic.keys()):
                        gt_bboxes = np.asarray(gt_bboxes_dic[class_name], dtype = np.float32)

                        gt_class = CLASS_DIC[class_name]
                        all_ground_truths_dic[class_name] += gt_bboxes.shape[0]

                        pred_bboxes = encode_bboxes[i, :, :]
                        pred_classes = encode_classes[i, :, gt_class][..., np.newaxis]
                        pred_bboxes = np.concatenate((pred_bboxes, pred_classes), axis = -1)

                        pred_bboxes[:, :4] = convert_bboxes(pred_bboxes[:, :4], img_wh = batch_image_wh[i])
                        pred_bboxes = nms(pred_bboxes, nms_threshold)

                        ious = compute_bboxes_IoU(pred_bboxes, gt_bboxes)

                        # ious >= 0.50 (AP@50)
                        correct = np.max(ious, axis = 1) >= ap_threshold
                        confidence = pred_bboxes[:, 4]

                        correct_dic[class_name] += correct.tolist()
                        confidence_dic[class_name] += confidence.tolist()

                batch_image_data = []
                batch_image_wh = []
                batch_gt_bboxes_dic = []

            sys.stdout.write('\r# Validation = {:.2f}%'.format(valid_iter / valid_xml_count * 100))
            sys.stdout.flush()

        if len(batch_image_data) != 0:
            encode_bboxes, encode_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : batch_image_data})

            for i in range(len(batch_image_data)):
                gt_bboxes_dic = batch_gt_bboxes_dic[i]
                for class_name in list(gt_bboxes_dic.keys()):
                    gt_bboxes = np.asarray(gt_bboxes_dic[class_name], dtype = np.float32)

                    gt_class = CLASS_DIC[class_name]
                    all_ground_truths_dic[class_name] += gt_bboxes.shape[0]

                    pred_bboxes = encode_bboxes[i, :, :]
                    pred_classes = encode_classes[i, :, gt_class][..., np.newaxis]
                    pred_bboxes = np.concatenate((pred_bboxes, pred_classes), axis = -1)

                    pred_bboxes[:, :4] = convert_bboxes(pred_bboxes[:, :4], img_wh = batch_image_wh[i])
                    pred_bboxes = nms(pred_bboxes, nms_threshold)

                    ious = compute_bboxes_IoU(pred_bboxes, gt_bboxes)

                    # ious >= 0.50 (AP@50)
                    correct = np.max(ious, axis = 1) >= ap_threshold
                    confidence = pred_bboxes[:, 4]

                    correct_dic[class_name] += correct.tolist()
                    confidence_dic[class_name] += confidence.tolist()

        valid_time = int(time.time() - valid_time)
        print('\n[i] valid time = {}sec'.format(valid_time))

        valid_mAP_list = []
        for class_name in CLASS_NAMES:
            if all_ground_truths_dic[class_name] == 0:
                continue

            all_ground_truths = all_ground_truths_dic[class_name]
            
            correct_list = correct_dic[class_name]
            confidence_list = confidence_dic[class_name]

            # list -> numpy
            confidence_list = np.asarray(confidence_list, dtype = np.float32)
            correct_list = np.asarray(correct_list, dtype = np.bool)
            
            # Ascending (confidence)
            sort_indexs = confidence_list.argsort()[::-1]
            confidence_list = confidence_list[sort_indexs]
            correct_list = correct_list[sort_indexs]
            
            correct_detections = 0
            all_detections = 0

            # calculate precision/recall
            precision_list = []
            recall_list = []

            for confidence, correct in zip(confidence_list, correct_list):
                all_detections += 1
                if correct:
                    correct_detections += 1    
                
                precision = correct_detections / all_detections
                recall = correct_detections / all_ground_truths
                
                precision_list.append(precision)
                recall_list.append(recall)

                # maximum correct detections
                if recall == 1.0:
                    break

            precision_list = np.asarray(precision_list, dtype = np.float32)
            recall_list = np.asarray(recall_list, dtype = np.float32)
            
            # calculating the interpolation performed in 11 points (0.0 -> 1.0, +0.01)
            precision_interp_list = []
            interp_list = np.arange(0, 10 + 1) / 10

            for interp in interp_list:
                try:
                    precision_interp = max(precision_list[recall_list >= interp])
                except:
                    precision_interp = 0.0
                
                precision_interp_list.append(precision_interp)

            ap = np.mean(precision_interp_list) * 100
            valid_mAP_list.append(ap)

        valid_mAP = np.mean(valid_mAP_list)
        if best_valid_mAP < valid_mAP:
            best_valid_mAP = valid_mAP
            saver.save(sess, './model/DSSD_{}.ckpt'.format(iter))
            
        log_print('[i] valid mAP : {:.6f}, best valid mAP : {:.6f}'.format(valid_mAP, best_valid_mAP))

saver.save(sess, './model/DSSD.ckpt')
