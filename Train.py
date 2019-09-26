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
train_data_list = np.load('./dataset/train_detection.npy', allow_pickle = True)[:100]
valid_data_list = np.load('./dataset/validation_detection.npy', allow_pickle = True)
valid_count = len(valid_data_list)

open('log.txt', 'w')
log_print('[i] Train : {}'.format(len(train_data_list)))
log_print('[i] Valid : {}'.format(len(valid_data_list)))

# 2. build
input_var = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
is_training = tf.placeholder(tf.bool)

fcos_dic, fcos_sizes = FCOS(input_var, is_training)
fcos_utils = FCOS_Utils(fcos_sizes)

pred_bboxes_op = fcos_dic['pred_bboxes']
pred_centers_op = fcos_dic['pred_centers']
pred_classes_op = fcos_dic['pred_classes']

log_print('[i] pred_bboxes_op : {}'.format(pred_bboxes_op))
log_print('[i] pred_centers_op : {}'.format(pred_centers_op))
log_print('[i] pred_classes_op : {}'.format(pred_classes_op))

_, fcos_size, _ = pred_bboxes_op.shape.as_list()
gt_bboxes_var = tf.placeholder(tf.float32, [BATCH_SIZE, fcos_size, 4])
gt_centers_var = tf.placeholder(tf.float32, [BATCH_SIZE, fcos_size, 1])
gt_classes_var = tf.placeholder(tf.float32, [BATCH_SIZE, fcos_size, CLASSES])

log_print('[i] gt_bboxes_var : {}'.format(gt_bboxes_var))
log_print('[i] gt_centers_var : {}'.format(gt_centers_var))
log_print('[i] gt_classes_var : {}'.format(gt_classes_var))

pred_ops = [pred_bboxes_op, pred_centers_op, pred_classes_op]
gt_ops = [gt_bboxes_var, gt_centers_var, gt_classes_var]
loss_op, focal_loss_op, center_loss_op, giou_loss_op = FCOS_Loss(pred_ops, gt_ops)

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY
loss_op += l2_reg_loss_op

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    # train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op)
    train_op = tf.train.MomentumOptimizer(learning_rate_var, momentum = 0.9).minimize(loss_op)

train_summary_dic = {
    'Total_Loss' : loss_op,
    'Focal_Loss' : focal_loss_op,
    'Center_Loss' : center_loss_op,
    'GIoU_Loss' : giou_loss_op,
    'L2_Regularization_Loss' : l2_reg_loss_op,
    'Learning_rate' : learning_rate_var,
}

train_summary_list = []
for name in train_summary_dic.keys():
    value = train_summary_dic[name]
    train_summary_list.append(tf.summary.scalar(name, value))
train_summary_op = tf.summary.merge(train_summary_list)

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

saver = tf.train.Saver(max_to_keep = 100)
# saver.restore(sess, './model/FCOS_{}.ckpt'.format(115000))

learning_rate = INIT_LEARNING_RATE

log_print('[i] max_iteration : {}'.format(MAX_ITERATION))
log_print('[i] decay_iteration : {}'.format(DECAY_ITERATIONS))

loss_list = []
focal_loss_list = []
center_loss_list = []
giou_loss_list = []
l2_reg_loss_list = []
train_time = time.time()

train_writer = tf.summary.FileWriter('./logs/train')

train_threads = []
for i in range(NUM_THREADS):
    train_thread = Teacher('./dataset/train_detection.npy', fcos_sizes, debug = False)
    train_thread.start()
    train_threads.append(train_thread)

for iter in range(1, MAX_ITERATION + 1):
    if iter in DECAY_ITERATIONS:
        learning_rate /= 10
        log_print('[i] learning rate decay : {} -> {}'.format(learning_rate * 10, learning_rate))

    # Thread
    find = False
    while not find:
        for train_thread in train_threads:
            if train_thread.ready:
                find = True
                batch_image_data, batch_encode_bboxes, batch_encode_centers, batch_encode_classes = train_thread.get_batch_data()        
                break
    
    _feed_dict = {input_var : batch_image_data, gt_bboxes_var : batch_encode_bboxes, gt_centers_var : batch_encode_centers, gt_classes_var : batch_encode_classes, 
                  is_training : True, learning_rate_var : learning_rate}
    log = sess.run([train_op, loss_op, focal_loss_op, center_loss_op, giou_loss_op, l2_reg_loss_op, train_summary_op], feed_dict = _feed_dict)
    # print(log[1:-1])
    
    if np.isnan(log[1]):
        print('[!]', log[1:-1])
        input()

    loss_list.append(log[1])
    focal_loss_list.append(log[2])
    center_loss_list.append(log[3])
    giou_loss_list.append(log[4])
    l2_reg_loss_list.append(log[5])
    train_writer.add_summary(log[6], iter)

    if iter % LOG_ITERATION == 0:
        loss = np.mean(loss_list)
        focal_loss = np.mean(focal_loss_list)
        center_loss = np.mean(center_loss_list)
        giou_loss = np.mean(giou_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] iter : {}, loss : {:.4f}, focal_loss : {:.4f}, center_loss : {:.4f}, giou_loss : {:.4f}, l2_reg_loss : {:.4f}, train_time : {}sec'.format(iter, loss, focal_loss, center_loss, giou_loss, l2_reg_loss, train_time))

        loss_list = []
        focal_loss_list = []
        center_loss_list = []
        giou_loss_list = []
        l2_reg_loss_list = []
        train_time = time.time()

    if iter % SAVE_ITERATION == 0:
        saver.save(sess, './model/FCOS_{}.ckpt'.format(iter))