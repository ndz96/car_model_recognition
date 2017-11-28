from car_data import *
from utils import *
from squeezenet import squeeze_net
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np

# Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# Prepare data
batch_size = 16
num_classes = 2
img_size = 227
num_channels = 3

data = CarData(batch_size)
data.load(img_size)

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, ], name='y_true')
y_pre_one_hot = tf.cast(y_true, tf.uint8)

# TO BE USED
y_one_hot = tf.one_hot(y_pre_one_hot, num_classes)
y_true_cls = tf.argmax(y_one_hot, axis = 1)

# xz, yz = data.get_train_batch()
# print(session.run([y_one_hot, y_true_cls], feed_dict = {y_true: yz}))

# Network graph params
# filter_size_conv1 = 3 
# num_filters_conv1 = 32

# filter_size_conv2 = 3
# num_filters_conv2 = 32

# filter_size_conv3 = 3
# num_filters_conv3 = 64

# fc_layer_size = 128

# layer_conv1 = create_convolutional_layer(input=x,
#                num_input_channels=num_channels,
#                conv_filter_size=filter_size_conv1,
#                num_filters=num_filters_conv1)
# layer_conv2 = create_convolutional_layer(input=layer_conv1,
#                num_input_channels=num_filters_conv1,
#                conv_filter_size=filter_size_conv2,
#                num_filters=num_filters_conv2)
# layer_conv3= create_convolutional_layer(input=layer_conv2,
#                num_input_channels=num_filters_conv2,
#                conv_filter_size=filter_size_conv3,
#                num_filters=num_filters_conv3)
          
# layer_flat = create_flatten_layer(layer_conv3)

# layer_fc1 = create_fc_layer(input=layer_flat,
#                      num_inputs=layer_flat.get_shape()[1:4].num_elements(),
#                      num_outputs=fc_layer_size,
#                      use_relu=True)

# layer_fc2 = create_fc_layer(input=layer_fc1,
#                      num_inputs=fc_layer_size,
#                      num_outputs=num_classes,
#                      use_relu=False) 

logits = squeeze_net(x, num_classes, tf.constant(0.5, dtype=tf.float32))
# print(logits.get_shape())

y_pred = tf.nn.softmax(logits,name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)

session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_one_hot)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.MomentumOptimizer(0.1, 0.9).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer())
num_epochs = 10
acc_train = 0

## TRAIN

num_steps = data.get_train_size() // batch_size

for i in range(num_epochs * num_steps):

    x_batch, y_batch = data.get_train_batch()

    feed_dict_tr = {x : x_batch,
                    y_true : y_batch}
    
    session.run(optimizer, feed_dict = feed_dict_tr)
    acc_train += session.run(accuracy, feed_dict = feed_dict_tr)

    if (i % 20 == 0):
        print("Train acc: ", acc_train / ((i % num_steps) + 1))

    if i % num_steps == 0:
        ## VALIDATION
        
        acc_val = 0
        num_steps_val = data.get_val_size() // batch_size
        for j in range(num_steps_val):
            x_val_batch, y_val_batch = data.get_val_batch()
            feed_dict_val = {x : x_val_batch,
                             y_true : y_val_batch}
            acc_val += session.run(accuracy, feed_dict = feed_dict_val)
        acc_val /= num_steps_val
        print("Epoch ", i//num_steps, ": Train acc: ", acc_train / num_steps, " Val acc: ", acc_val)
        acc_train = 0





