# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

from Define import *
from Utils import *
from DataAugmentation import *

class FCOS_Utils:
    def __init__(self, sizes):
        self.sizes = sizes
        self.generate_centers()

    def generate_centers(self):
        self.centers = {}
        self.decode_centers = []

        for level, size in zip(PYRAMID_LEVELS, self.sizes):
            w, h = size
            
            cx = np.arange(w) + 0.5
            cy = np.arange(h) + 0.5
            cx, cy = np.meshgrid(cx, cy)

            centers = np.concatenate([cx[..., np.newaxis], cy[..., np.newaxis]], axis = -1)
            centers = centers / [w, h] * [IMAGE_WIDTH, IMAGE_HEIGHT]

            self.centers['P%d'%level] = centers
            self.decode_centers.append(centers.reshape((-1, 2)))

        self.decode_centers = np.concatenate(self.decode_centers, axis = 0)

    def Encode(self, gt_bboxes, gt_classes):
        # 1. prepare gt_bboxes (bboxes + classes) & pyramid_dictionary
        gt_bboxes = np.concatenate([gt_bboxes, gt_classes[:, np.newaxis]], axis = -1)

        # 2. generate gt_bboxes, gt_classes and gt_centers.
        total_encode_bboxes = []
        total_encode_centers = []
        total_encode_classes = []
        
        for i in range(len(self.sizes)):
            w, h = self.sizes[i]
            pyramid_name = 'P{}'.format(PYRAMID_LEVELS[i])
            
            # get separate bboxes & centers
            centers = self.centers[pyramid_name].reshape((-1, 2))
            
            # create encode_bboxes, centers and classes.
            encode_bboxes = np.zeros((h * w, 4), dtype = np.float32)
            encode_centers = np.zeros((h * w, 1), dtype = np.float32)
            encode_classes = np.zeros((h * w, CLASSES), dtype = np.float32)

            for bbox in gt_bboxes:
                xmin, ymin, xmax, ymax, c = bbox

                # in center_x, center_y
                x_mask = np.logical_and(xmin <= centers[:, 0], centers[:, 0] <= xmax)
                y_mask = np.logical_and(ymin <= centers[:, 1], centers[:, 1] <= ymax)
                in_mask = np.logical_and(x_mask, y_mask)
                
                # calculate l*, t*, r*, b*
                l = np.maximum(centers[:, 0] - xmin, 0)
                t = np.maximum(centers[:, 1] - ymin, 0)
                r = np.maximum(xmax - centers[:, 0], 0)
                b = np.maximum(ymax - centers[:, 1], 0)
                ltrb = np.stack([l, t, r, b]).T
                
                max_v = np.max(ltrb, axis = -1)
                max_mask = np.logical_and(max_v >= M_LIST[i], max_v <= M_LIST[i + 1])

                # calculate center-ness (0 to 1)
                center_ness = (np.minimum(l, r) * np.minimum(t, b)) / (np.maximum(l, r) * np.maximum(t, b))
                center_ness = np.sqrt(center_ness)

                # in_mask, higher than center-ness
                mask = np.logical_and(in_mask, encode_centers[:, 0] < center_ness)
                mask = np.logical_and(mask, max_mask)

                # update
                encode_bboxes[mask, 0] = l[mask]
                encode_bboxes[mask, 1] = t[mask]
                encode_bboxes[mask, 2] = r[mask]
                encode_bboxes[mask, 3] = b[mask]
                encode_centers[mask, 0] = center_ness[mask]
                encode_classes[mask, :] = one_hot(c)

            # stack
            total_encode_bboxes.append(encode_bboxes)
            total_encode_centers.append(encode_centers)
            total_encode_classes.append(encode_classes)

        # concatenation
        total_encode_bboxes = np.concatenate(total_encode_bboxes, axis = 0)
        total_encode_centers = np.concatenate(total_encode_centers, axis = 0)
        total_encode_classes = np.concatenate(total_encode_classes, axis = 0)

        return total_encode_bboxes, total_encode_centers, total_encode_classes

    def Decode(self, encode_bboxes, encode_centers, encode_classes, image_wh, detect_threshold = 0.05, use_nms = False, topk = 100):
        # 1. get Regression (left, top, right, bottom)
        lt = self.decode_centers - encode_bboxes[:, :2]
        rb = self.decode_centers + encode_bboxes[:, 2:]
        
        decode_bboxes = np.concatenate([lt, rb], axis = -1)

        # 2. 
        class_probs = np.max(encode_classes[:, 1:], axis = -1)
        class_indexs = np.argmax(encode_classes[:, 1:], axis = -1)
        
        # with center-ness
        # class_probs *= encode_centers[:, 0]

        topk_prob = np.sort(class_probs)[::-1][topk]
        cond = np.logical_and(class_probs >= detect_threshold, topk_prob <= class_probs)
        
        pred_bboxes = convert_bboxes(decode_bboxes[cond], image_wh = image_wh)
        class_probs = class_probs[cond][..., np.newaxis]
        
        pred_bboxes = np.concatenate((pred_bboxes, class_probs), axis = -1)
        pred_classes = class_indexs[cond] + 1

        if use_nms:
            pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes)

        return pred_bboxes.astype(np.float32), pred_classes.astype(np.int32)


