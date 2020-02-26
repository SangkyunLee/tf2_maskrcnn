#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 15:59:09 2020

@author: cv_test
"""


def conv_block(kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    
    # kernel_size = 3
    # filters = [64, 64, 256]
    # stage=2
    # block='a'
    # strides=(1, 1)
    # use_bias=True
    # train_bn=True
    #########################
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    
    layers=[]
    shc_layers=[]
    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)#(input_tensor)
    layers.append(x)
    x = BatchNorm(name=bn_name_base + '2a')#(x, training=train_bn)
    layers.append(x)
    x = KL.Activation('relu')#(x)
    layers.append(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias) #x
    layers.append(x)
    x = BatchNorm(name=bn_name_base + '2b') #(x, training=train_bn)
    layers.append(x)
    x = KL.Activation('relu')#(x)
    layers.append(x)
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)#(x)
    layers.append(x)
    x = BatchNorm(name=bn_name_base + '2c')#(x, training=train_bn)
    layers.append(x)
    
    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)#(input_tensor)
    shc_layers.append(shortcut)
    shortcut = BatchNorm(name=bn_name_base + '1')#(shortcut, training=train_bn)
    shc_layers.append(shortcut)
    
    olayer = []
    x = KL.Add()#([x, shortcut])
    olayer.append(x)
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')#(x)
    olayer.append(x)
    return layers, shc_layers, olayer



def identity_block(kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    layer =[]
    

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)#(input_tensor)
    layer.append(x)
    x = BatchNorm(name=bn_name_base + '2a')#(x, training=train_bn)
    layer.append(x)
    
    x = KL.Activation('relu')#(x)
    layer.append(x)
    
    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)#(x)
    layer.append(x)
    
    x = BatchNorm(name=bn_name_base + '2b')#(x, training=train_bn)
    layer.append(x)
    
    x = KL.Activation('relu')#(x)
    layer.append(x)
    
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)#(x)
    layer.append(x)
    
    x = BatchNorm(name=bn_name_base + '2c')#(x, training=train_bn)
    layer.append(x)
    
    olayer =[]
    x = KL.Add()#([x, input_tensor])
    olayer.append(x)
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')#(x)
    olayer.append(x)
    return layer, olayer


 x=conv_block( 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)