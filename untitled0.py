# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 07:53:07 2019

@author: Sang
"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K

class SL(KL.Layer):
    
    def __init__(self, proposal_count, **kwargs):
        
        self.proposal_count = proposal_count
        super(SL, self).__init__(**kwargs)
    
    def build(self, input_shape):
       # assert isinstance(input_shape, list)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][2], 100),
                                      initializer='uniform',
                                      trainable=True)
        super(SL, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self,inputs):
        a, b = inputs
        out = K.dot(a, self.kernel) + b
        padding = tf.maximum(self.proposal_count - tf.shape(out)[1], 0)
        out = tf.pad(out, [(0,0),(0, padding), (0, 0)])
        return out
    
    def compute_output_shape(self, input_shape):
        print(self.proposal_count)
        return (None, self.proposal_count, 100)
    

test_input = [KL.Input(shape=[5, 10]),KL.Input(shape=[5, 100])]

f = SL(proposal_count=100)
fout =f(test_input)
dim = f.output_shape

###############################33

import numpy as np


a1 = tf.constant(np.random.randint(0,2,(2,5)))

def tsum(input):
    return tf.math.reduce_sum(input)
def map_tsum(input):
    return tf.map_fn(tsum,input)

out =map_tsum(a1)



