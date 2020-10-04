#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:00:07 2020

@author: root
"""
import tensorflow as tf
from mrcnn_tf2 import model_utils as mutil
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import numpy as np
# inputs = val_inputs


def compare(subTestmodel, subTrainmodel):
    out ={}
   
    if hasattr(subTestmodel,'layers') and hasattr(subTrainmodel,'layers'):
        testlayers = subTestmodel.layers
        trainlayers = subTrainmodel.layers
        for testl, trainl in zip(testlayers, trainlayers):
            if isinstance(testl,KM.Model) and isinstance(trainl,KM.Model):
                print(testl)
                out1 = compare(testl, trainl)
                out.update(out1)
            else:    
                if testl.__class__.__name__ == 'TimeDistributed':
                    Wte=testl.layer.get_weights()
                    Wtr=trainl.layer.get_weights()
                else:
                    Wte = testl.get_weights()
                    Wtr = trainl.get_weights()
                sumv = 0
                for Wte_, Wtr_ in zip(Wte, Wtr):
                     sumv += tf.reduce_sum(tf.abs(Wte_-Wtr_)).numpy()
                out[testl.name] = sumv

    return out

compare(tmodel, model)

for bn in range(1,5):    
    tlayer = getattr(tmodel.layers[-1], 'bn%d'%bn)
    mlayer = getattr(model.layers[-1], 'bn%d'%bn)
    W = mlayer.layer.get_weights()
    tlayer.layer.set_weights(W)
    
    # lenw = len(tlayer.layer.weights)
    # for i in range(lenw):
    #     tlayer.layer.weights[i] = 0*mlayer.layer.weights[i] 




@tf.function
def validate_trainmodel(model, inputs):
    input_image = inputs[0]
    input_image_meta = inputs[1]
    config = model.get_config()
    
    # out = model(inputs[:5])
    # mrcnn_class = out[4]  
    # mrcnn_bbox = out[5]

    #input_image, input_image_meta, windows = model.mold_inputs([original_image])
    #outputs = tmodel.predict([input_image, input_image_meta])
    #outputs = tmodel([input_image, input_image_meta])
    
    out_fpn = model.fpn(input_image)   
    rpn_feature_maps = out_fpn
    mrcnn_feature_maps = out_fpn[:-1]
    
    layer_outputs = []  # list of lists
    for p in rpn_feature_maps:
        layer_outputs.append(model.rpn(p))
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))    

    
    outputs = [KL.Concatenate(axis=1, name=n)(list(o))                   
                for o, n in zip(outputs, output_names)]
    
    rpn_class_logits, rpn_class, rpn_bbox = outputs
    
    anchors = model.get_anchors(config.IMAGE_SHAPE)
    anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
    
    # print('PL_start')
    rpn_rois = model.PL([rpn_class, rpn_bbox, anchors])
    
    
    rois = rpn_rois
    #rois = model.target_rois # the output of detectionTargetLayer
    
    inputs_fpn =[rois, input_image_meta, *mrcnn_feature_maps]    
    # class, bbox        
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = model.fpn_clf(inputs_fpn)
    
    
    mrcnn_feature_maps = out_fpn[:-1]    
    DL = mutil.DetectionLayer(config, name="mrcnn_detection")    
    inputs_dl =[rois,  mrcnn_class, mrcnn_bbox, input_image_meta ]
    detections = DL(inputs_dl)
    
    
    detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
    inputs_fpn =[detection_boxes, input_image_meta, *mrcnn_feature_maps]  
    mrcnn_mask = model.fpn_mask(inputs_fpn)    
    
    detections = detections.numpy()
    mrcnn_mask = mrcnn_mask.numpy()
  
   # Process detections
    results = []
    for i in range(input_image.shape[0]):
        
        final_rois, final_class_ids, final_scores, final_masks =\
            model.unmold_detections(detections[i], mrcnn_mask[i],\
                                    input_image[i].shape, input_image[i].shape,\
                                    windows[i])
        results.append({
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }) 
    return results



results = validate_trainmodel(tmodel, val_inputs)
idx=[0,1]
r = results[1]
visualize.display_instances(original_image, r['rois'][idx], r['masks'][:,:,idx], r['class_ids'][idx], 
                            dataset_val.class_names, r['scores'][idx], figsize=(8, 8))