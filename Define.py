# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

# dataset parameters
ROOT_DIR = 'D:/_ImageDataset/'

CLASS_NAMES = ['background'] + ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}
CLASSES = len(CLASS_NAMES)

# network parameters
IMAGE_HEIGHT = 800
IMAGE_WIDTH = 1024
IMAGE_CHANNEL = 3

# m2 = 0, m3 = 64, ..., m7 = inf (=1024)
M_LIST = [-1, -1, 0, 64, 128, 256, 512, 1024]

AP_THRESHOLD = 0.5
NMS_THRESHOLD = 0.6

# loss parameters
WEIGHT_DECAY = 0.0001

# train
BATCH_SIZE = 8
INIT_LEARNING_RATE = 1e-4

MAX_EPOCH = 200
LOG_ITERATION = 50
VALID_ITERATION = 5000
