import tensorflow as tf
import numpy as np

def squeeze_net(images, num_classes, dropout):
    net = {}

    net['input'] = images
    sq_ratio = 1

    # conv1_1
    net['conv1'] = conv_layer('conv1', net['input'],
                            W=weight_variable(shape = [3, 3, 3, 64], name='conv1_w'), stride=[1, 2, 2, 1])

    net['relu1'] = relu_layer('relu1', net['conv1'], b=bias_variable([64], 'relu1_b'))
    net['pool1'] = pool_layer('pool1', net['relu1'])

    net['fire2'] = fire_module('fire2', net['pool1'], sq_ratio * 16, 64, 64)
    net['fire3'] = fire_module('fire3', net['fire2'], sq_ratio * 16, 64, 64,   True)
    net['pool3'] = pool_layer('pool3', net['fire3'])

    net['fire4'] = fire_module('fire4', net['pool3'], sq_ratio * 32, 128, 128)
    net['fire5'] = fire_module('fire5', net['fire4'], sq_ratio * 32, 128, 128, True)
    net['pool5'] = pool_layer('pool5', net['fire5'])

    net['fire6'] = fire_module('fire6', net['pool5'], sq_ratio * 48, 192, 192)
    net['fire7'] = fire_module('fire7', net['fire6'], sq_ratio * 48, 192, 192, True)
    net['fire8'] = fire_module('fire8', net['fire7'], sq_ratio * 64, 256, 256)
    net['fire9'] = fire_module('fire9', net['fire8'], sq_ratio * 64, 256, 256, True)

    # 50% dropout
    net['dropout9'] = tf.nn.dropout(net['fire9'], dropout)
    net['conv10'] = conv_layer('conv10', net['dropout9'],
                            W=weight_variable([1, 1, 512, num_classes], name='conv10', init='normal'))
    net['relu10'] = relu_layer('relu10', net['conv10'], b=bias_variable([num_classes], 'relu10_b'))
    net['pool10'] = pool_layer('pool10', net['relu10'], pooling_type='avg')

    logits = tf.reshape(net['pool10'], [-1, num_classes])
    return logits

def bias_variable( shape, name, value=0.1):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial)

def weight_variable( shape, name=None, init='xavier'):
    if init == 'variance':
        initial = tf.get_variable('W' + name, shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    elif init == 'xavier':
        initial = tf.get_variable('W' + name, shape, initializer=tf.contrib.layers.xavier_initializer())
    else:
        initial = tf.Variable(tf.random_normal(shape, stddev=0.01), name='W'+name)

    return initial

def relu_layer( layer_name, layer_input, b=None):
    if b:
        layer_input += b
    relu = tf.nn.relu(layer_input)
    return relu

def pool_layer( layer_name, layer_input, pooling_type='max'):
    if pooling_type == 'avg':
        pool = tf.nn.avg_pool(layer_input, ksize=[1, 13, 13, 1],
                            strides=[1, 1, 1, 1], padding='VALID')
    elif pooling_type == 'max':
        pool = tf.nn.max_pool(layer_input, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='VALID')
    return pool

def conv_layer( layer_name, layer_input, W, stride=[1, 1, 1, 1]):
    return tf.nn.conv2d(layer_input, W, strides=stride, padding='SAME')

def fire_module( layer_name, layer_input, s1x1, e1x1, e3x3, residual=False):
    """ Fire module consists of squeeze and expand convolutional layers. """
    fire = {}

    shape = layer_input.get_shape()

    # squeeze
    s1_weight = weight_variable([1, 1, int(shape[3]), s1x1], layer_name + '_s1')

    # expand
    e1_weight = weight_variable([1, 1, s1x1, e1x1], layer_name + '_e1')
    e3_weight = weight_variable([3, 3, s1x1, e3x3], layer_name + '_e3')

    fire['s1'] = conv_layer(layer_name + '_s1', layer_input, W=s1_weight)
    fire['relu1'] = relu_layer(layer_name + '_relu1', fire['s1'],
                                    b=bias_variable([s1x1], layer_name + '_fire_bias_s1'))

    fire['e1'] = conv_layer(layer_name + '_e1', fire['relu1'], W=e1_weight)
    fire['e3'] = conv_layer(layer_name + '_e3', fire['relu1'], W=e3_weight)
    fire['concat'] = tf.concat([tf.add(fire['e1'], bias_variable([e1x1],
                                                        name=layer_name + '_fire_bias_e1' )),
                                tf.add(fire['e3'], bias_variable([e3x3],
                                                        name=layer_name + '_fire_bias_e3' ))], 3)

    if residual:
        fire['relu2'] = relu_layer(layer_name + 'relu2_res', tf.add(fire['concat'],layer_input))
    else:
        fire['relu2'] = relu_layer(layer_name + '_relu2', fire['concat'])

    return fire['relu2']
