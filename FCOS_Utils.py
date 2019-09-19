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

        m_indexs = np.arange(3, 7 + 1)

        for index, size in zip(m_indexs, self.sizes):
            w, h = size
            decode_bboxes = np.zeros([h, w, 4], dtype = np.float32)
            decode_classes = np.zeros([h, w, CLASSES], dtype = np.float32)

            decode_classes[:, :, 0] = 1.

            for gt_bbox, gt_class in zip(gt_bboxes, gt_classes):
                xmin, ymin, xmax, ymax = gt_bbox
                
                center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
                width, height = xmax - xmin, ymax - ymin

                bbox_size = max(width, height)
                # print(bbox_size, M_LIST[index - 1], M_LIST[index])

                if bbox_size >= M_LIST[index - 1] and bbox_size <= M_LIST[index]:
                    grid_x = int(center_x / IMAGE_WIDTH * w)
                    grid_y = int(center_y / IMAGE_HEIGHT * h)

                    decode_bboxes[grid_y, grid_x, :] = gt_bbox
                    decode_classes[grid_y, grid_x, :] = one_hot(gt_class)

            decode_bboxes = decode_bboxes.reshape((-1, 4))
            decode_classes = decode_classes.reshape((-1, CLASSES))

            total_decode_bboxes = np.append(total_decode_bboxes, decode_bboxes, axis = 0)
            total_decode_classes = np.append(total_decode_classes, decode_classes, axis = 0)

        return total_decode_bboxes, total_decode_classes

    def Encode_Debug(self, gt_bboxes, gt_classes):
        info_list = []

        total_decode_bboxes = np.zeros((0, 4), dtype = np.float32)
        total_decode_classes = np.zeros((0, CLASSES), dtype = np.float32)

        m_indexs = np.arange(3, 7 + 1)

        for index, size in zip(m_indexs, self.sizes):
            w, h = size
            decode_bboxes = np.zeros([h, w, 4], dtype = np.float32)
            decode_classes = np.zeros([h, w, CLASSES], dtype = np.float32)

            decode_classes[:, :, 0] = 1.

            for gt_bbox, gt_class in zip(gt_bboxes, gt_classes):
                xmin, ymin, xmax, ymax = gt_bbox
                
                center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
                width, height = xmax - xmin, ymax - ymin

                bbox_size = max(width, height)
                # print(bbox_size, M_LIST[index - 1], M_LIST[index])

                if bbox_size >= M_LIST[index - 1] and bbox_size <= M_LIST[index]:
                    grid_x = int(center_x / IMAGE_WIDTH * w)
                    grid_y = int(center_y / IMAGE_HEIGHT * h)

                    grid_cx = (grid_x + 0.5) / w * IMAGE_WIDTH
                    grid_cy = (grid_y + 0.5) / h * IMAGE_HEIGHT

                    l = grid_cx - xmin
                    t = grid_cy - ymin
                    r = xmax - grid_cx
                    b = ymax - grid_cy

                    lr_center_ness = min(l, r) / max(max(l, r), 1e-12)
                    tb_center_ness = min(t, b) / max(max(t, b), 1e-12)
                    center_ness = np.sqrt(lr_center_ness * tb_center_ness)

                    info_list.append([grid_cx, grid_cy, l, t, r, b, center_ness])
                    
                    decode_bboxes[grid_y, grid_x, :] = gt_bbox
                    decode_classes[grid_y, grid_x, :] = one_hot(gt_class)

            decode_bboxes = decode_bboxes.reshape((-1, 4))
            decode_classes = decode_classes.reshape((-1, CLASSES))

            total_decode_bboxes = np.append(total_decode_bboxes, decode_bboxes, axis = 0)
            total_decode_classes = np.append(total_decode_classes, decode_classes, axis = 0)

        return total_decode_bboxes, total_decode_classes, info_list

    def Decode(self, decode_bboxes, decode_classes, image_wh, detect_threshold = 0.05, use_nms = False):
        total_class_probs = np.max(decode_classes[:, 1:], axis = -1)
        total_class_indexs = np.argmax(decode_classes[:, 1:], axis = -1)

        cond = total_class_probs >= detect_threshold
            
        pred_bboxes = decode_bboxes[cond]
        class_probs = total_class_probs[cond][..., np.newaxis]
        
        pred_bboxes = np.concatenate((pred_bboxes, class_probs), axis = -1)
        pred_classes = total_class_indexs[cond] + 1

        pred_bboxes[:, :4] = convert_bboxes(pred_bboxes[:, :4], image_wh = image_wh)

        if use_nms:
            pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes)

        return pred_bboxes.astype(np.float32), pred_classes.astype(np.int32)

if __name__ == '__main__':
    import cv2
    from FCOS import *

    input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    fcos_dic, fcos_sizes = FCOS(input_var, False)
    
    # 1. Test GT bboxes (Encode -> Decode)
    fcos_utils = FCOS_Utils(fcos_sizes)
    xml_paths = glob.glob('D:/DB/VOC2007/train/xml/*.xml')
    
    for xml_path in xml_paths:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = True)

        image = cv2.imread(image_path)
        h, w, c = image.shape
        
        decode_bboxes, decode_classes, info_list = fcos_utils.Encode_Debug(gt_bboxes, gt_classes)
        positive_count = np.sum(decode_classes[:, 1:])
        print(positive_count, decode_classes.shape)
        
        pred_bboxes, pred_classes = fcos_utils.Decode(decode_bboxes, decode_classes, [w, h], use_nms = True)
        print(pred_bboxes.shape, pred_classes.shape)

        # center info
        for info in info_list:
            data = np.asarray(info[:-1]).reshape((-1, 2)) / [IMAGE_WIDTH, IMAGE_HEIGHT] * [w, h]
            grid_cx, grid_cy, l, t, r, b = data.reshape(-1).astype(np.int32)

            center_ness = info[-1]
            
            cv2.arrowedLine(image, (grid_cx, grid_cy), (grid_cx - l, grid_cy), (255, 0, 128), 1)
            cv2.arrowedLine(image, (grid_cx, grid_cy), (grid_cx + r, grid_cy), (255, 0, 128), 1)
            cv2.arrowedLine(image, (grid_cx, grid_cy), (grid_cx, grid_cy - t), (255, 0, 128), 1)
            cv2.arrowedLine(image, (grid_cx, grid_cy), (grid_cx, grid_cy + b), (255, 0, 128), 1)
            cv2.circle(image, (grid_cx, grid_cy), 1, (0, 0, 255), 2)

            cv2.putText(image, 'center = {:.2f}'.format(center_ness), (grid_cx - 10, grid_cy - 10), 1, 1, (0, 0, 255), 2)
        
        for pred_bbox, pred_class in zip(pred_bboxes, pred_classes):
            xmin, ymin, xmax, ymax = pred_bbox[:4].astype(np.int32)
            confidence = pred_bbox[4] * 100
            class_name = CLASS_NAMES[pred_class]

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

