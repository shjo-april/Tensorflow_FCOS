# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

# dataset parameters
ROOT_DIR = 'D:/_ImageDataset/COCO/'
TRAIN_DIR = ROOT_DIR + 'train2017/image/'
VALID_DIR = ROOT_DIR + 'valid2017/image/'

CLASS_NAMES = ['background'] + [class_name.strip() for class_name in open('./coco/label_names.txt').readlines()]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}
CLASSES = len(CLASS_NAMES)

# network parameters
IMAGE_HEIGHT = 800
IMAGE_WIDTH = 1024
IMAGE_CHANNEL = 3

# ResNetv2-50 (Normalize), OpenCV BGR -> RGB
R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94
MEAN = [R_MEAN, G_MEAN, B_MEAN]

# Feature Pyramid
PYRAMID_LEVELS = [3, 4, 5, 6, 7]
STRIDES = [8, 16, 32, 64, 128]

# m2 = 0, m3 = 64, ..., m7 = inf (=1024)
M_LIST = [0, 64, 128, 256, 512, 1024]

AP_THRESHOLD = 0.5
NMS_THRESHOLD = 0.6

# loss parameters
WEIGHT_DECAY = 0.0001

# train

# multi gpu training
# GPU_INFO = "0,1,2,3"

# single gpu training
GPU_INFO = "0"

NUM_GPU = len(GPU_INFO.split(','))
BATCH_SIZE = 2 * NUM_GPU
INIT_LEARNING_RATE = 0.01

# SGD with momentum
# Learning Rate = 0.01, Batch Size = 16, momentum = 0.9, weight decay = 0.0001

# Adam (Leanring rate 0.0001)
# Learning Rate = 0.001, Batch Size = 16, momentum = 0.9, weight decay = 0.0001

# use thread (Dataset)
NUM_THREADS = 10

# iteration & learning rate schedule
MAX_ITERATION = 90000
DECAY_ITERATIONS = [60000, 80000]

LOG_ITERATION = 50
SAVE_ITERATION = 5000
VALID_ITERATION = 5000

# color_list (OpenCV - BGR)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

COLOR_PBLUE = (204, 72, 63)
COLOR_ORANGE = (0, 128, 255)
