import os
import cv2
import glob

import numpy as np
import tensorflow as tf

from FCOS import *
from FCOS_Utils import *

##############################################################################################
# prepare FCOS !
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
fcos_dic, fcos_sizes = FCOS(input_var, False)

fcos_utils = FCOS_Utils(fcos_sizes)
##############################################################################################

##############################################################################################
# 1. Test Check Centers
# for level, size in zip(PYRAMID_LEVELS, fcos_utils.sizes):
#     w, h = size

#     centers = fcos_utils.centers['P%d'%level]
#     bg = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.uint8)

#     for y in range(h):
#         for x in range(w):
#             cx, cy = (centers[y, x] / [w, h] * [IMAGE_WIDTH, IMAGE_HEIGHT]).astype(np.int32)
#             cv2.circle(bg, (cx, cy), 1, (0, 255, 0), 2)

#     cv2.imshow('show', bg)
#     cv2.waitKey(0)
##############################################################################################

##############################################################################################
# 2. Test GT bboxes
# for data in np.load('./dataset/train_detection.npy', allow_pickle = True):
#     # 2.0 load image and labels.
#     image_name, gt_bboxes, gt_classes = data

#     image_path = TRAIN_DIR + image_name
#     image = cv2.imread(image_path)
#     h, w, c = image.shape

#     gt_bboxes = np.asarray(gt_bboxes, dtype = np.float32)
#     gt_bboxes = gt_bboxes / [w, h, w, h] * [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]

#     gt_classes = np.asarray([CLASS_DIC[c] for c in gt_classes], dtype = np.int32)

#     # 2.1 original show
#     image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
#     for gt_bbox in gt_bboxes:
#         xmin, ymin, xmax, ymax = gt_bbox.astype(np.int32)
#         cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#     cv2.imshow('original', image)
#     cv2.waitKey(1)
    
#     # 2.2 pyramid_dic
#     gt_bboxes = np.concatenate([gt_bboxes, gt_classes[:, np.newaxis]], axis = -1)
#     pyramid_dic = {'P%d'%level : [] for level in PYRAMID_LEVELS}

#     for i in range(len(fcos_utils.sizes)):    
#         m_index = i + 1
#         pyramid_name = 'P{}'.format(PYRAMID_LEVELS[i])

#         for gt_bbox in gt_bboxes:
#             # get width, height
#             width = gt_bbox[2] - gt_bbox[0]
#             height = gt_bbox[3] - gt_bbox[1]

#             # calculate bbox_area = root(width * height)
#             bbox_area = np.sqrt(width * height)
#             if bbox_area >= M_LIST[m_index - 1] and bbox_area <= M_LIST[m_index]:
#                 pyramid_dic[pyramid_name].append(gt_bbox)

#     # 2.3 pyramid 
#     for i in range(len(fcos_utils.sizes)):
#         w, h = fcos_utils.sizes[i]
#         pyramid_name = 'P{}'.format(PYRAMID_LEVELS[i])
        
#         # get separate bboxes & centers
#         bboxes = np.asarray(pyramid_dic[pyramid_name], dtype = np.float32)
#         centers = fcos_utils.centers[pyramid_name] / fcos_utils.sizes[i] * [IMAGE_WIDTH, IMAGE_HEIGHT]
#         centers = centers.reshape((-1, 2))

#         # create encode_bboxes, centers and classes.
#         encode_bboxes = np.zeros((h * w, 4), dtype = np.float32)
#         encode_centers = np.zeros((h * w, 1), dtype = np.float32)
#         encode_classes = np.zeros((h * w, CLASSES), dtype = np.float32)

#         # set background
#         encode_classes[:, 0] = 1.

#         # calculate l*, t*, r*, b*, center-ness
#         for bbox in bboxes:
#             xmin, ymin, xmax, ymax, c = bbox

#             # in center_x, center_y
#             x_mask = np.logical_and(xmin <= centers[:, 0], centers[:, 0] <= xmax)
#             y_mask = np.logical_and(ymin <= centers[:, 1], centers[:, 1] <= ymax)
#             in_mask = np.logical_and(x_mask, y_mask)
            
#             # calculate l*, t*, r*, b*
#             l = np.maximum(centers[:, 0] - xmin, 0)
#             t = np.maximum(centers[:, 1] - ymin, 0)
#             r = np.maximum(xmax - centers[:, 0], 0)
#             b = np.maximum(ymax - centers[:, 1], 0)
            
#             # calculate center-ness (0 to 1)
#             center_ness = (np.minimum(l, r) * np.minimum(t, b)) / (np.maximum(l, r) * np.maximum(t, b))
#             center_ness = np.sqrt(center_ness)

#             # in_mask, higher than center-ness
#             mask = np.logical_and(in_mask, encode_centers[:, 0] < center_ness)
            
#             # update
#             encode_bboxes[mask, 0] = l[mask]
#             encode_bboxes[mask, 1] = t[mask]
#             encode_bboxes[mask, 2] = r[mask]
#             encode_bboxes[mask, 3] = b[mask]
#             encode_centers[mask, 0] = center_ness[mask]
#             encode_classes[mask, :] = one_hot(c)

#         # reshape
#         centers = centers.reshape((h, w, 2))
#         encode_bboxes = encode_bboxes.reshape((h, w, 4))
#         encode_centers = encode_centers.reshape((h, w, 1))
#         encode_classes = encode_classes.reshape((h, w, CLASSES))
        
#         bg = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.uint8)

#         # show gt_bboxes
#         for bbox in bboxes:
#             xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
#             cv2.rectangle(bg, (xmin, ymin), (xmax, ymax), COLOR_PBLUE, 1)
        
#         # show centers (circle, in = orange, out = green)
#         for y in range(h):
#             for x in range(w):
#                 cx, cy = centers[y, x].astype(np.int32)
                
#                 if np.argmax(encode_classes[y, x]) != 0:
#                     l, t, r, b = encode_bboxes[y, x, :].astype(np.int32)

#                     cv2.arrowedLine(bg, (cx, cy), (cx - l, cy), COLOR_ORANGE, 1, tipLength = 0.05)
#                     cv2.arrowedLine(bg, (cx, cy), (cx + r, cy), COLOR_ORANGE, 1, tipLength = 0.05)
#                     cv2.arrowedLine(bg, (cx, cy), (cx, cy - t), COLOR_ORANGE, 1, tipLength = 0.05)
#                     cv2.arrowedLine(bg, (cx, cy), (cx, cy + b), COLOR_ORANGE, 1, tipLength = 0.05)

#                     cv2.circle(bg, (cx, cy), 1, COLOR_ORANGE, 2)
#                 else:
#                     cv2.circle(bg, (cx, cy), 1, COLOR_GREEN, 1)
        
#         cv2.imshow('show', bg)
#         cv2.waitKey(0)
##############################################################################################

##############################################################################################
# 3. final Encode & Decode Test
for data in np.load('./dataset/train_detection.npy', allow_pickle = True):
    image_name, gt_bboxes, gt_classes = data

    image_path = TRAIN_DIR + image_name
    image = cv2.imread(image_path)
    h, w, c = image.shape

    gt_bboxes = np.asarray(gt_bboxes, dtype = np.float32)
    gt_bboxes = gt_bboxes / [w, h, w, h] * [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]
    
    gt_classes = np.asarray([CLASS_DIC[c] for c in gt_classes], dtype = np.int32)
    
    # Encode
    encode_bboxes, encode_centers, encode_classes = fcos_utils.Encode(gt_bboxes, gt_classes)
    
    # print(encode_bboxes.shape, np.min(encode_bboxes), np.max(encode_bboxes))
    # print(encode_centers.shape, np.min(encode_centers), np.max(encode_centers))
    # print(encode_classes.shape, np.sum(encode_classes[:, 1:]), len(gt_bboxes))
    # input()

    # Decode
    pred_bboxes, pred_classes = fcos_utils.Decode(encode_bboxes, encode_centers, encode_classes, [IMAGE_WIDTH, IMAGE_HEIGHT], )

    # Show
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    for pred_bbox, pred_class in zip(pred_bboxes, pred_classes):
        xmin, ymin, xmax, ymax = pred_bbox[:4].astype(np.int32)
        conf = pred_bbox[4]

        cv2.putText(image, CLASS_NAMES[pred_class], (xmin, ymin - 10), 1, 1, COLOR_GREEN, 2)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR_GREEN, 2)

    cv2.imshow('show', image)
    cv2.waitKey(0)

##############################################################################################

