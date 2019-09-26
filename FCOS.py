# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

import resnet_v2.resnet_v2 as resnet_v2

from Define import *

kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01, seed = None)
bias_initializer = tf.constant_initializer(value = 0.0)
class_bias_initializer = tf.constant_initializer(value = -np.log((1 - 0.01) / 0.01))

def group_normalization(x, is_training, G = 32, ESP = 1e-5, scope = 'group_norm'):
    with tf.variable_scope(scope):
        # 1. [N, H, W, C] -> [N, C, H, W]
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.shape.as_list()

        # 2. reshape (group normalization)
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        
        # 3. get mean, variance
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        # 4. normalize
        x = (x - mean) / tf.sqrt(var + ESP)

        # 5. create gamma, bete
        gamma = tf.Variable(tf.constant(1.0, shape = [C]), dtype = tf.float32, name = 'gamma')
        beta = tf.Variable(tf.constant(0.0, shape = [C]), dtype = tf.float32, name = 'beta')

        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        
        # 6. gamma * x + beta
        x = tf.reshape(x, [-1, C, H, W]) * gamma + beta

        # 7. [N, C, H, W] -> [N, H, W, C]
        x = tf.transpose(x, [0, 2, 3, 1])
    return x

def conv_gn_relu(x, filters, kernel_size, strides, padding, is_training, scope, gn = True, activation = True, use_bias = True, upscaling = False):
    with tf.variable_scope(scope):
        if not upscaling:
            x = tf.layers.conv2d(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = kernel_initializer, use_bias = use_bias, name = 'conv2d')
        else:
            x = tf.layers.conv2d_transpose(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = kernel_initializer, use_bias = use_bias, name = 'upconv2d')
        
        if gn:
            x = group_normalization(x, is_training = is_training, scope = 'gn')
        
        if activation:
            x = tf.nn.relu(x, name = 'relu')
    return x

def connection_block(x1, x2, is_training, scope):
    with tf.variable_scope(scope):
        x1 = conv_gn_relu(x1, 256, [3, 3], 1, 'same', is_training, 'conv1', gn = True, activation = False)
        x2 = conv_gn_relu(x2, 256, [1, 1], 1, 'valid', is_training, 'conv2', gn = True, activation = False)
        x = tf.nn.relu(x1 + x2, name = 'relu')
    return x

def build_head_loc(x, is_training, name, depth = 4):
    with tf.variable_scope(name):
        for i in range(depth):
            x = conv_gn_relu(x, 256, (3, 3), 1, 'same', is_training, '{}'.format(i))
        
        center_ness_x = conv_gn_relu(x, 1, (3, 3), 1, 'same', is_training, 'center-ness', gn = False, activation = False) 
        regression_x = conv_gn_relu(x, 4, (3, 3), 1, 'same', is_training, 'regression', gn = False, activation = False)
    return regression_x, center_ness_x

def build_head_cls(x, is_training, name, depth = 4):
    with tf.variable_scope(name):
        for i in range(depth):
            x = conv_gn_relu(x, 256, (3, 3), 1, 'same', is_training, '{}'.format(i))
        
        x = tf.layers.conv2d(inputs = x, filters = CLASSES, kernel_size = [3, 3], strides = 1, padding = 'same', 
                             kernel_initializer = kernel_initializer, bias_initializer = class_bias_initializer, name = 'classification')
    return x

def FCOS_Decode_Layer(pred_bboxes, decode_centers):
    # calculate xmin, ymin, xmax, ymax
    xmin = decode_centers[..., 0] - pred_bboxes[..., 0] 
    ymin = decode_centers[..., 1] - pred_bboxes[..., 1]
    xmax = decode_centers[..., 0] + pred_bboxes[..., 2]
    ymax = decode_centers[..., 1] + pred_bboxes[..., 3]
    
    xmin = tf.clip_by_value(xmin[..., tf.newaxis], 0, IMAGE_WIDTH - 1)
    ymin = tf.clip_by_value(ymin[..., tf.newaxis], 0, IMAGE_HEIGHT - 1)
    xmax = tf.clip_by_value(xmax[..., tf.newaxis], 0, IMAGE_WIDTH - 1)
    ymax = tf.clip_by_value(ymax[..., tf.newaxis], 0, IMAGE_HEIGHT - 1)
    
    # concatenate bboxes (xmin, ymin, xmax, ymax)
    pred_bboxes = tf.concat([xmin, ymin, xmax, ymax], axis = -1)
    return pred_bboxes

def FCOS_ResNet_50(input_var, is_training, reuse = False):
    # convert BGR -> RGB
    x = input_var[..., ::-1]

    # ResNetv2-50 (ImageNet)
    x -= MEAN
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_50(x, is_training = is_training, reuse = reuse)
    
    # for key in end_points.keys():
    #     print(key, end_points[key])
    # input()

    pyramid_dic = {}
    feature_maps = [end_points['resnet_v2_50/block{}'.format(i)] for i in [4, 2, 1]]

    pyramid_dic['C3'] = feature_maps[2]
    pyramid_dic['C4'] = feature_maps[1]
    pyramid_dic['C5'] = feature_maps[0]

    # print(pyramid_dic['C3'])
    # print(pyramid_dic['C4'])
    # print(pyramid_dic['C5'])

    fcos_dic = {}
    fcos_sizes = []
    
    with tf.variable_scope('FCOS', reuse = reuse):
        x = conv_gn_relu(pyramid_dic['C5'], 256, (1, 1), 1, 'valid', is_training, 'P5_conv')
        pyramid_dic['P5'] = x

        x = conv_gn_relu(x, 256, (3, 3), 2, 'same', is_training, 'P6_conv')
        pyramid_dic['P6'] = x
        
        x = conv_gn_relu(x, 256, (3, 3), 2, 'same', is_training, 'P7_conv')
        pyramid_dic['P7'] = x

        x = conv_gn_relu(pyramid_dic['P5'], 256, (3, 3), 2, 'same', is_training, 'P4_conv_1', upscaling = True)
        x = connection_block(x, pyramid_dic['C4'], is_training, 'P4_conv')
        pyramid_dic['P4'] = x

        x = conv_gn_relu(pyramid_dic['P4'], 256, (3, 3), 2, 'same', is_training, 'P3_conv_1', upscaling = True)
        x = connection_block(x, pyramid_dic['C3'], is_training, 'P3_conv')
        pyramid_dic['P3'] = x
        
        '''
        # P3 : Tensor("FCOS/add_1:0", shape=(8, 100, 128, 256), dtype=float32)
        # P4 : Tensor("FCOS/add:0", shape=(8, 50, 64, 256), dtype=float32)
        # P5 : Tensor("FCOS/P5_conv/relu:0", shape=(8, 25, 32, 256), dtype=float32)
        # P6 : Tensor("FCOS/P6_conv/relu:0", shape=(8, 13, 16, 256), dtype=float32)
        # P7 : Tensor("FCOS/P7_conv/relu:0", shape=(8, 7, 8, 256), dtype=float32)
        '''
        # for i in range(3, 7 + 1):
        #    print('# P{} :'.format(i), pyramid_dic['P{}'.format(i)])
        # input()
        
        pred_bboxes = []
        pred_centers = []
        pred_classes = []
        
        for i in range(len(PYRAMID_LEVELS)):
            level = PYRAMID_LEVELS[i]
            feature_map = pyramid_dic['P{}'.format(level)]
            _, h, w, c = feature_map.shape.as_list()
            
            # get regression, center-ness, classification
            _pred_bboxes, _pred_centers = build_head_loc(feature_map, is_training, 'P{}_Regression_Branch'.format(level))
            _pred_classes = build_head_cls(feature_map, is_training, 'P{}_Classification_Branch'.format(level))

            # activation
            _pred_bboxes = tf.exp(_pred_bboxes)
            _pred_classes = tf.nn.sigmoid(_pred_classes)
            
            # parsing (l*, t*, r*, b*)
            l = tf.clip_by_value(_pred_bboxes[:, :, :, 0] * STRIDES[i], 0, IMAGE_WIDTH)
            t = tf.clip_by_value(_pred_bboxes[:, :, :, 1] * STRIDES[i], 0, IMAGE_HEIGHT)
            r = tf.clip_by_value(_pred_bboxes[:, :, :, 2] * STRIDES[i], 0, IMAGE_WIDTH)
            b = tf.clip_by_value(_pred_bboxes[:, :, :, 3] * STRIDES[i], 0, IMAGE_HEIGHT)

            # merge l*, t*, r*, b*
            _pred_bboxes = tf.transpose(tf.stack([l, t, r, b]), [1, 2, 3, 0])

            # reshape bboxes, classes, center-ness
            _pred_bboxes = tf.reshape(_pred_bboxes, [-1, h * w, 4])
            _pred_centers = tf.reshape(_pred_centers, [-1, h * w, 1])
            _pred_classes = tf.reshape(_pred_classes, [-1, h * w, CLASSES])
            
            # append sizes, bboxes, centers, classes
            fcos_sizes.append([w, h])
            pred_bboxes.append(_pred_bboxes)
            pred_centers.append(_pred_centers)
            pred_classes.append(_pred_classes)

        # concatenate bboxes, centers, classes (axis = 1)
        pred_bboxes = tf.concat(pred_bboxes, axis = 1, name = 'Regression')
        pred_centers = tf.concat(pred_centers, axis = 1, name = 'Center-ness')
        pred_classes = tf.concat(pred_classes, axis = 1, name = 'Classification')

        # update dictionary 
        fcos_dic['pred_bboxes'] = pred_bboxes
        fcos_dic['pred_centers'] = pred_centers
        fcos_dic['pred_classes'] = pred_classes

    return fcos_dic, fcos_sizes

FCOS = FCOS_ResNet_50

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [8, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    
    fcos_dic, fcos_sizes = FCOS(input_var, False)
    
    print(fcos_dic['pred_bboxes'])
    print(fcos_dic['pred_centers'])
    print(fcos_dic['pred_classes'])
