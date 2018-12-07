#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:50:14 2018

@author: pbchandr
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:55:29 2018

@author: pbchandr
"""

import tensorflow as tf
import numpy as np
import sys

class Cnn(object):
    """
    A CNN for gene expression prediction.
    Uses a convolution layer, max-pooling, and softmax layer with dropouts
    """

    def __init__(self, sequence_width, sequence_height, num_classes, num_conv_filters, 
                 conv_filter_sizes, conv_strides, pooling_size, num_fc_layers, num_fc_kernels,
                 lrn_rate=0.001, l2_reg_lambda=0.0):
         
        print 'Running Cnn'
        
        # Parse conv filters, fiter sizes, and strides
        num_conv_filters = [int(x) for x in num_conv_filters.split(',')]
        conv_filter_sizes = [int(x) for x in conv_filter_sizes.split(',')]
        conv_strides = [int(x) for x in conv_strides.split(',')]
        
        num_fc_kernels = [int(x) for x in num_fc_kernels.split(',')]
        
        # Dimension checks
        if len(num_conv_filters) != len(conv_filter_sizes) | len(num_conv_filters) != len(conv_strides):
            print("ERROR: Filter and stride arrays should have equal size")
            sys.exit(1)
            
        for _, stride in enumerate(conv_strides):
            if sequence_width/stride < pooling_size:
                print("ERROR: Pooling size should be less than the convolution dimension. Choose pooling size less than %s"%int(sequence_width/stride))
                sys.exit(1)  
        
        if len(num_fc_kernels) != num_fc_layers:
            print("ERROR: Number of layers and number of kernels do not match")
            sys.exit(1)
            
        
        # Placeholders for input data, class labels, and dropouts
        self.input_x = tf.placeholder(tf.int32, [None, sequence_width, sequence_height], name="input_x")
        print('Input shape: ', self.input_x.shape)
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        print('Class label shape: ', self.input_y.shape)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        # Input_x is 3D. We need to expand it to 4D to apply convolution
        self.input_x_expanded = tf.expand_dims(self.input_x, -1)
        self.input_x_expanded = tf.cast(self.input_x_expanded, tf.float32)
        print('Input expand shape: ', self.input_x_expanded.shape)
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        # Convolution layer with max pooling
        pooled_outputs = []
        total_nodes = 0
        for indx, filter_size in enumerate(conv_filter_sizes):
            with tf.name_scope("conv_maxpool-%s" %filter_size):
                # Convolution layer
                filter_shape = [filter_size, sequence_height, 1, num_conv_filters[indx]]
                print("filter_sape: ", filter_shape)
                conv_weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv-wt")
                print("conv weight shape: ", conv_weight.shape)
                conv_bias = tf.Variable(tf.constant(0.1, shape=[num_conv_filters[indx]]), name="conv-bias")
                print("conv bias shape: ", conv_bias.shape)
                conv = tf.nn.conv2d(self.input_x_expanded, conv_weight, 
                                    strides=[1, conv_strides[indx], 1, 1], 
                                    padding="VALID", name="conv")
                print("conv: ", conv)
                
                # Apply activation aka non-linearity
                conv_relu = tf.nn.relu(tf.nn.bias_add(conv, conv_bias), name="relu")
                print("relu: ", conv_relu)
                
                # Pooling - Max pooling over the given pooling size
                conv_pool = tf.nn.max_pool(conv_relu, ksize=[1, pooling_size, 1, 1],
                                                    strides=[1, pooling_size, 1, 1], 
                                                    padding='VALID', name="pool")
                print('conv_pool', conv_pool)
                num_nodes = np.ceil((sequence_width - filter_size)/pooling_size)
                total_nodes = total_nodes + int(num_nodes*num_conv_filters[indx])
                print('Total Nodes', total_nodes)
                pooled_outputs.append(conv_pool)
         
        # Combine all the pooled features
        h_pool = tf.concat(pooled_outputs, 2)
        print('hpool shape: ', h_pool.shape)
        self.h_pool_flat = tf.reshape(h_pool, [-1, total_nodes])
        print('hpool shape: ', self.h_pool_flat.shape)
        
        # Add Dropout layer
        with tf.variable_scope('dropout'):
            dropout_layer = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            
        # Fully connected layers
        with tf.variable_scope('fully_connected_layers'):
            fc_layer = []
            for i in range(num_fc_layers):
                if i == 0:
                    n_1 = total_nodes
                    n_2 = num_fc_kernels[i]
                    fcn_ip_layer = dropout_layer
                else:
                    n_1 = num_fc_kernels[(i-1)]
                    n_2 = num_fc_kernels[i]
                    fcn_ip_layer = fc_layer
                    
                fc_weight = tf.Variable(tf.truncated_normal([n_1, n_2], stddev = 0.1))
                fc_bias = tf.Variable(tf.zeros([n_2]))
                fc_layer = tf.nn.relu(tf.matmul(fcn_ip_layer, fc_weight) + fc_bias)
                print('fc_layer', fc_layer)
        
        # Output
        with tf.name_scope("output"):
            output_weight = tf.get_variable(shape=[num_fc_kernels[-1], num_classes],
                                            initializer=tf.contrib.layers.xavier_initializer(),
                                            name="out_wt")
            output_bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="out_b")
            l2_loss += tf.nn.l2_loss(output_weight)
            l2_loss += tf.nn.l2_loss(output_bias)
            self.logit = tf.nn.xw_plus_b(fc_layer, output_weight, output_bias, name="logit")
            self.predictions = tf.argmax(self.logit, 1, name="predictions")
        
        # Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.cost = self.loss

        # Optimization
        with tf.name_scope("optimize"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lrn_rate).minimize(self.loss)
        
            