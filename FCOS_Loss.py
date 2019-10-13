# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import tensorflow as tf

from Define import *

'''
pt = {
    p    , if y = 1
    1 − p, otherwise
}
FL(pt) = −(1 − pt)γ * log(pt)
'''
def Focal_Loss(pred_classes, gt_classes, alpha = 0.25, gamma = 2):
    with tf.variable_scope('Focal'):
        # focal_loss = [BATCH_SIZE, 22890, CLASSES]
        pt = gt_classes * pred_classes + (1 - gt_classes) * (1 - pred_classes) 
        focal_loss = -alpha * tf.pow(1. - pt, gamma) * tf.log(pt + 1e-10)

        # focal_loss = [BATCH_SIZE]
        focal_loss = tf.reduce_sum(tf.abs(focal_loss), axis = [1, 2])
    
    return focal_loss

'''
GIoU = IoU - (C - (A U B))/C
Loss = 1 - GIoU
'''
def GIoU(pred_bboxes, gt_bboxes):
    with tf.variable_scope('GIoU'):
        # 1. unstack (pred_bboxes, gt_bboxes)
        pred_l, pred_t, pred_r, pred_b = tf.unstack(pred_bboxes, axis = -1)
        gt_l, gt_t, gt_r, gt_b = tf.unstack(gt_bboxes, axis = -1)

        # 2. calulate intersection over union
        pred_area = (pred_l + pred_r) * (pred_t + pred_b)
        gt_area = (gt_l + gt_r) * (gt_t + gt_b)

        inter_w = tf.minimum(pred_l, gt_l) + tf.minimum(pred_r, gt_r)
        inter_h = tf.minimum(pred_t, gt_t) + tf.minimum(pred_b, gt_b)
        
        inter = tf.maximum(inter_w * inter_h, 0.)
        union = pred_area + gt_area - inter
        
        ious = inter / tf.maximum(union, 1e-10)

        # 3. (C - (A U B))/C
        C_w = tf.maximum(pred_l, gt_l) + tf.maximum(pred_r, gt_r)
        C_h = tf.maximum(pred_t, gt_t) + tf.maximum(pred_b, gt_b)
        C = tf.maximum(C_w * C_h, 0.)
        
        giou = ious - (C - union) / tf.maximum(C, 1e-10)
    return giou

def FCOS_Loss(pred_ops, gt_ops, alpha = 1.0):
    # parsing
    pred_bboxes, pred_centers, pred_classes = pred_ops
    gt_bboxes, gt_centers, gt_classes = gt_ops

    # positive_mask = [BATCH_SIZE, 22890]
    positive_mask = tf.reduce_max(gt_classes[:, :, 1:], axis = -1)
    positive_mask = tf.cast(tf.math.equal(positive_mask, 1.), dtype = tf.float32)
    
    # positive_count = [BATCH_SIZE]
    positive_count = tf.reduce_sum(positive_mask, axis = 1)
    positive_count = tf.clip_by_value(positive_count, 1, positive_count)

    # calculate focal_loss & center-ness_loss & GIoU_loss
    focal_loss_op = Focal_Loss(pred_classes, gt_classes)

    center_loss_op = tf.nn.sigmoid_cross_entropy_with_logits(logits = pred_centers, labels = gt_centers)
    center_loss_op = tf.reduce_sum(center_loss_op, axis = [1, 2])
    
    giou_loss_op = 1 - GIoU(pred_bboxes, gt_bboxes)
    giou_loss_op = tf.reduce_sum(positive_mask * giou_loss_op, axis = 1)
    
    # divide positive_count
    focal_loss_op = tf.reduce_mean(focal_loss_op / positive_count)
    center_loss_op = tf.reduce_mean(center_loss_op / positive_count)
    giou_loss_op = tf.reduce_mean(giou_loss_op / positive_count)
    
    # final loss functions
    loss_op = focal_loss_op + center_loss_op + alpha * giou_loss_op
    
    return loss_op, focal_loss_op, center_loss_op, giou_loss_op

if __name__ == '__main__':
    pred_bboxes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, 4])
    pred_centers = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, 1])
    pred_classes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, CLASSES])
    
    gt_bboxes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, 4])
    gt_centers = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, 1])
    gt_classes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, CLASSES])

    pred_ops = [pred_bboxes, pred_centers, pred_classes]
    gt_ops = [gt_bboxes, gt_centers, gt_classes]
    
    loss_op, focal_loss_op, center_loss_op, giou_loss_op = FCOS_Loss(pred_ops, gt_ops)
    print(loss_op, focal_loss_op, center_loss_op, giou_loss_op)

