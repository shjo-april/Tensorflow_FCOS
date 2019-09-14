
import cv2
import time
import random
import threading

import numpy as np

from Define import *
from Utils import *
from DataAugmentation import *
from SSD_Utils import *

class Teacher(threading.Thread):
    ready = False
    min_data_size = 0
    max_data_size = 50

    anchors = []
    xml_paths = []

    batch_data_list = []
    batch_data_length = 0

    debug = False
    name = ''

    def __init__(self, xml_paths, anchors, min_data_size = 1, max_data_size = 50, name = 'Thread', debug = False):
        self.min_data_size = min_data_size
        self.max_data_size = max_data_size

        self.xml_paths = xml_paths
        self.name = name
        self.anchors = anchors.copy()
        self.debug = debug

        threading.Thread.__init__(self)
        
    def get_batch_data(self):
        batch_image_data, batch_gt_bboxes, batch_gt_classes = self.batch_data_list[0]
        
        del self.batch_data_list[0]
        self.batch_data_length -= 1

        if self.batch_data_length < self.min_data_size:
            self.ready = False
        
        return batch_image_data, batch_gt_bboxes, batch_gt_classes
    
    def run(self):
        while True:
            while self.batch_data_length >= self.max_data_size:
                continue

            batch_image_data = []
            batch_gt_bboxes = []
            batch_gt_classes = []
            batch_xml_paths = random.sample(self.xml_paths, BATCH_SIZE * 2)

            for xml_path in batch_xml_paths:
                if self.debug:
                    delay = time.time()
                
                image, gt_bboxes, gt_classes = get_data(xml_path, training = True)

                if self.debug:
                    delay = time.time() - delay
                    print('[D] {} - {} = {}ms'.format(self.name, 'xml', int(delay * 1000)))

                encode_bboxes, encode_classes = Encode(gt_bboxes, gt_classes, self.anchors)

                batch_image_data.append(image.astype(np.float32))
                batch_gt_bboxes.append(encode_bboxes)
                batch_gt_classes.append(encode_classes)
            
            batch_image_data = np.asarray(batch_image_data, dtype = np.float32) 
            batch_gt_bboxes = np.asarray(batch_gt_bboxes, dtype = np.float32)
            batch_gt_classes = np.asarray(batch_gt_classes, dtype = np.float32)
            
            self.batch_data_list.append([batch_image_data, batch_gt_bboxes, batch_gt_classes])
            self.batch_data_length += 1

            if self.batch_data_length >= self.min_data_size:
                self.ready = True
            else:
                self.ready = False
