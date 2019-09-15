import numpy as np
import tensorflow as tf

import resnet_v2.resnet_v2 as resnet_v2

from Define import *

initializer = tf.contrib.layers.xavier_initializer()

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

def conv_bn_relu(x, filters, kernel_size, strides, padding, is_training, scope, gn = True, activation = True, use_bias = True, upscaling = False):
    with tf.variable_scope(scope):
        if not upscaling:
            x = tf.layers.conv2d(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = initializer, use_bias = use_bias, name = 'conv2d')
        else:
            x = tf.layers.conv2d_transpose(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = initializer, use_bias = use_bias, name = 'upconv2d')
        
        if gn:
            x = group_normalization(x, is_training = is_training, scope = 'gn')

        if activation:
            x = tf.nn.relu(x, name = 'relu')
    return x

def connection_block(x1, x2, is_training, scope):
    with tf.variable_scope(scope):
        x1 = conv_bn_relu(x1, 256, [3, 3], 1, 'same', is_training, 'conv1', gn = True, activation = False)
        x2 = conv_bn_relu(x2, 256, [1, 1], 1, 'valid', is_training, 'conv2', gn = True, activation = False)

        x = tf.nn.relu(x1 + x2, name = 'relu')
    return x

def build_head_loc(x, is_training, name, depth = 4):
    with tf.variable_scope(name):
        for i in range(depth):
            x = conv_bn_relu(x, 256, (3, 3), 1, 'same', is_training, '{}'.format(i))
        x = conv_bn_relu(x, 4, (3, 3), 1, 'same', is_training, 'loc', gn = False, activation = False)
    return x

def build_head_cls(x, is_training, name, depth = 4):
    with tf.variable_scope(name):
        for i in range(depth):
            x = conv_bn_relu(x, 256, (3, 3), 1, 'same', is_training, '{}'.format(i))
        x = conv_bn_relu(x, CLASSES, (3, 3), 1, 'same', is_training, 'cls', gn = False, activation = False)
    return x

def FCOS_ResNet_50(input_var, is_training, reuse = False):

    x = input_var - [103.939, 123.68, 116.779]
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
        # freeze : batch norm
        logits, end_points = resnet_v2.resnet_v2_50(x, is_training = False, reuse = reuse)
    
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
        x = conv_bn_relu(pyramid_dic['C5'], 256, (1, 1), 1, 'valid', is_training, 'P5_conv')
        pyramid_dic['P5'] = x

        x = conv_bn_relu(x, 256, (3, 3), 2, 'same', is_training, 'P6_conv')
        pyramid_dic['P6'] = x
        
        x = conv_bn_relu(x, 256, (3, 3), 2, 'same', is_training, 'P7_conv')
        pyramid_dic['P7'] = x

        x = conv_bn_relu(pyramid_dic['P5'], 256, (3, 3), 2, 'same', is_training, 'P4_conv_1', upscaling = True)
        x = connection_block(x, pyramid_dic['C4'], is_training, 'P4_conv')
        pyramid_dic['P4'] = x

        x = conv_bn_relu(pyramid_dic['P4'], 256, (3, 3), 2, 'same', is_training, 'P3_conv_1', upscaling = True)
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
        pred_classes = []
        
        for i in range(3, 7 + 1):
            feature_map = pyramid_dic['P{}'.format(i)]
            _, h, w, c = feature_map.shape.as_list()
            
            _pred_bboxes = build_head_loc(feature_map, is_training, 'P{}_bboxes'.format(i))
            _pred_classes = build_head_cls(feature_map, is_training, 'P{}_classes'.format(i))

            # parsing (l*, t*, r*, b*)
            # print('P{}'.format(i), M_LIST[i - 1], M_LIST[i])
            l = tf.clip_by_value(tf.exp(_pred_bboxes[:, :, :, 0]), M_LIST[i - 1], M_LIST[i])
            t = tf.clip_by_value(tf.exp(_pred_bboxes[:, :, :, 1]), M_LIST[i - 1], M_LIST[i])
            r = tf.clip_by_value(tf.exp(_pred_bboxes[:, :, :, 2]), M_LIST[i - 1], M_LIST[i])
            b = tf.clip_by_value(tf.exp(_pred_bboxes[:, :, :, 3]), M_LIST[i - 1], M_LIST[i])

            # generate centers (shape = [w, h, 2])
            xs = tf.range(w, dtype = tf.float32) + 0.5
            ys = tf.range(h, dtype = tf.float32) + 0.5

            xs, ys = tf.meshgrid(xs, ys)
            center_xys = tf.concat([xs[..., tf.newaxis], ys[..., tf.newaxis]], axis = -1)

            # calculate xmin, ymin, xmax, ymax
            xmin = center_xys[..., 0] - l
            ymin = center_xys[..., 1] - t
            xmax = center_xys[..., 0] + r
            ymax = center_xys[..., 1] + b

            xmin = tf.clip_by_value(xmin[..., tf.newaxis], 0, IMAGE_WIDTH - 1)
            ymin = tf.clip_by_value(ymin[..., tf.newaxis], 0, IMAGE_HEIGHT - 1)
            xmax = tf.clip_by_value(xmax[..., tf.newaxis], 0, IMAGE_WIDTH - 1)
            ymax = tf.clip_by_value(ymax[..., tf.newaxis], 0, IMAGE_HEIGHT - 1)

            # concatenate bboxes (xmin, ymin, xmax, ymax)
            _pred_bboxes = tf.concat([xmin, ymin, xmax, ymax], axis = -1)

            # reshape bboxes, classes (without center-ness)
            _pred_bboxes = tf.reshape(_pred_bboxes, [-1, h * w, 4])
            _pred_classes = tf.reshape(_pred_classes, [-1, h * w, CLASSES])
            
            # append sizes, bboxes, classes
            fcos_sizes.append([w, h])
            pred_bboxes.append(_pred_bboxes)
            pred_classes.append(_pred_classes)

        # concatenate bboxes, classes (axis = 1)
        pred_bboxes = tf.concat(pred_bboxes, axis = 1, name = 'bboxes')
        pred_classes = tf.concat(pred_classes, axis = 1, name = 'classes')

        # update dictionary 
        fcos_dic['pred_bboxes'] = pred_bboxes
        fcos_dic['pred_classes'] = tf.nn.sigmoid(pred_classes)

    return fcos_dic, fcos_sizes

FCOS = FCOS_ResNet_50

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [8, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    
    fcos_dic, fcos_sizes = FCOS(input_var, False)
    
    print(fcos_dic['pred_bboxes'])
    print(fcos_dic['pred_classes'])
