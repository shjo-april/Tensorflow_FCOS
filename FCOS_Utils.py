# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

from Define import *
from Utils import *
from DataAugmentation import *

def get_data(xml_path, training, normalize = True, augment = True):
    if training:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = False)

        image = cv2.imread(image_path)
        
        if augment:
            image, gt_bboxes, gt_classes = DataAugmentation(image, gt_bboxes, gt_classes)

        image_h, image_w, image_c = image.shape
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

        gt_bboxes = gt_bboxes.astype(np.float32)
        gt_classes = np.asarray(gt_classes, dtype = np.int32)

        if normalize:
            gt_bboxes /= [image_w, image_h, image_w, image_h]
            gt_bboxes *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]
    else:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = normalize)
        image = cv2.imread(image_path)

    return image, gt_bboxes, gt_classes

class FCOS_Utils:
    def __init__(self, sizes):
        self.sizes = sizes

    def Encode(self, gt_bboxes, gt_classes):
        total_decode_bboxes = np.zeros((0, 4), dtype = np.float32)
        total_decode_classes = np.zeros((0, CLASSES), dtype = np.float32)

        for size in self.sizes:
            w, h = size
            decode_bboxes = np.zeros([h, w, 4], dtype = np.float32)
            decode_classes = np.zeros([h, w, CLASSES], dtype = np.float32)

            decode_classes[:, :, 0] = 1.

            for gt_bbox, gt_class in zip(gt_bboxes, gt_classes):
                xmin, ymin, xmax, ymax = gt_bbox
                center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2

                grid_x = int(center_x / IMAGE_WIDTH * w)
                grid_y = int(center_y / IMAGE_HEIGHT * h)

                decode_bboxes[grid_y, grid_x, :] = gt_bbox
                decode_classes[grid_y, grid_x, :] = one_hot(gt_class)

            decode_bboxes = decode_bboxes.reshape((-1, 4))
            decode_classes = decode_classes.reshape((-1, CLASSES))

            total_decode_bboxes = np.append(total_decode_bboxes, decode_bboxes, axis = 0)
            total_decode_classes = np.append(total_decode_classes, decode_classes, axis = 0)

        return total_decode_bboxes, total_decode_classes

    def Decode(self, decode_bboxes, decode_classes, image_wh, find_class = None, detect_threshold = 0.05, use_nms = False):
        # pred_bboxes = [?, 6]
        if find_class is None:
            total_class_probs = np.max(decode_classes, axis = -1)
            total_class_indexs = np.argmax(decode_classes, axis = -1)

            cond = total_class_indexs > 0

            pred_bboxes = decode_bboxes[cond]
            class_probs = total_class_probs[cond][..., np.newaxis]
            class_indexs = total_class_indexs[cond][..., np.newaxis]
            
            pred_bboxes = np.concatenate((pred_bboxes, class_probs, class_indexs), axis = -1)
        # pred_bboxes = [?, 5]
        else:
            total_class_probs = decode_classes[:, find_class]
            
            cond = total_class_probs >= detect_threshold
            
            pred_bboxes = decode_bboxes[cond]
            class_probs = total_class_probs[cond][..., np.newaxis]

            pred_bboxes = np.concatenate((pred_bboxes, class_probs), axis = -1)

        pred_bboxes[:, :4] = convert_bboxes(pred_bboxes[:, :4], image_wh = image_wh)

        if use_nms:
            pred_bboxes = nms(pred_bboxes, NMS_THRESHOLD)

        return pred_bboxes

if __name__ == '__main__':
    import cv2
    from FCOS import *

    input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    fcos_dic, fcos_sizes = FCOS(input_var, False)
    
    # 1. Test GT bboxes (Encode -> Decode)
    fcos_utils = FCOS_Utils(fcos_sizes)
    xml_paths = glob.glob('D:/_DeepLearning_DB/VOC2007/train/xml/*.xml')
    
    for xml_path in xml_paths:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = True)

        image = cv2.imread(image_path)
        h, w, c = image.shape
        
        decode_bboxes, decode_classes = fcos_utils.Encode(gt_bboxes, gt_classes)
        positive_count = np.sum(decode_classes[:, 1:])
        print(positive_count, decode_classes.shape)

        pred_bboxes = fcos_utils.Decode(decode_bboxes, decode_classes, [w, h], use_nms = True)
        print(pred_bboxes.shape)
        
        for pred_bbox in pred_bboxes:
            xmin, ymin, xmax, ymax = pred_bbox[:4].astype(np.int32)
            confidence = pred_bbox[4] * 100
            class_name = CLASS_NAMES[int(pred_bbox[5])]

            cv2.putText(image, '{} = {:.2f}%'.format(class_name, confidence), (xmin, ymin - 10), 1, 1, (0, 255, 0), 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

        # class_name = 'person'
        # pred_bboxes = fcos_utils.Decode(decode_bboxes, decode_classes, [w, h], find_class = CLASS_DIC[class_name], use_nms = True)
        # print(pred_bboxes.shape)
        
        # for pred_bbox in pred_bboxes:
        #     xmin, ymin, xmax, ymax = pred_bbox[:4].astype(np.int32)
        #     confidence = pred_bbox[4] * 100

        #     cv2.putText(image, '{} = {:.2f}%'.format(class_name, confidence), (xmin, ymin - 10), 1, 1, (0, 255, 0), 1)
        #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

        cv2.imshow('show', image)
        cv2.waitKey(0)

