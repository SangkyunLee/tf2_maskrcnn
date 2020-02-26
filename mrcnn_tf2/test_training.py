
import tensorflow as tf
@tf.function
def compute_loss(model, inputs):
    # input_image = inputs[0]
    input_image_meta = inputs[1]
    input_rpn_match = inputs[2]
    input_rpn_bbox = inputs[3]
    
    # input_gt_class_ids = inputs[4]
    # input_gt_boxes = inputs[5]
    # input_gt_masks = inputs[6]
    
    config = model.get_config()
    
    out = model(inputs)
    
    rpn_class_logits = out[0]
    # rpn_class = out[1]
    rpn_bbox = out[2]
    
    # mrcnn_class_logits = out[3]
    # # mrcnn_class = out[4]  
    # mrcnn_bbox = out[5]
    # mrcnn_mask = out[6]
    
    # target_class_ids = out[9]
    # target_bbox = out[10]
    # target_mask = out[11]

    
    rpn_class_loss  = ML.rpn_class_loss_graph(input_rpn_match, rpn_class_logits)
    rpn_bbox_loss = ML.rpn_bbox_loss_graph(config, input_rpn_bbox, input_rpn_match, rpn_bbox)
    
    # active_class_ids =  mutil.parse_image_meta_graph(input_image_meta)['active_class_ids']
    # class_loss = ML.mrcnn_class_loss_graph(target_class_ids, mrcnn_class_logits, active_class_ids)  
    
     
    # bbox_loss = ML.mrcnn_bbox_loss_graph(target_bbox, target_class_ids, mrcnn_bbox)         
    # mask_loss = ML.mrcnn_mask_loss_graph(target_mask, target_class_ids, mrcnn_mask)
    
    
    
    f=tf.keras.regularizers.l2
    weight = config.WEIGHT_DECAY
        
    reg_losses =[f(weight)(w)/tf.cast(tf.size(w),tf.float32)
                  for w in  model.trainable_weights
                  if 'gamma' not in w.name and 'beta' not in w.name]
    
    main_loss =  rpn_class_loss + rpn_bbox_loss #+ class_loss + bbox_loss + mask_loss 
    main_loss += reg_losses
    
    return main_loss        
    

@tf.function
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss
  

from mrcnn_tf2 import data

learning_rate = 0.001
epochs=1
layers='heads'
augmentation=None
custom_callbacks=None
no_augmentation_sources=None
  

train_generator =data.data_generator(train_dataset, model.config, shuffle=True,                                 
                        augmentation=augmentation,
                        batch_size=model.config.BATCH_SIZE,
                        no_augmentation_sources=no_augmentation_sources)

layers='heads'

# Pre-defined layer regular expressions
layer_regex = {
    # all layers but the backbone
    "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
    # From a specific Resnet stage and up
    "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
    "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
    "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
    # All layers
    "all": ".*",
}
if layers in layer_regex.keys():
    layers_sel = layer_regex[layers]
    



# from tensorflow.contrib.memory_stats import BytesInUse
import tensorflow as tf
import tensorflow.keras.layers as KL


epochs=5
learning_rate = config.LEARNING_RATE
momentum =  config.LEARNING_MOMENTUM
optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                    clipnorm=config.GRADIENT_CLIP_NORM)

losses=[]
for epoch in range(epochs):
    for step, input_train in enumerate(train_generator):
        
        # input_train = next(train_generator)
        
        if step==0 and epoch ==0:
            ypred = model(input_train[0])                  
            model.set_trainable(layers_sel)  
            mutil.load_weights(model.fpn.resnet,'./mask_rcnn_coco.h5',by_name = True)
        
        loss = compute_apply_gradients(model, input_train[0], optimizer)
        print('epoch:{}. step {},  loss={}'.format(epoch, step,loss))

################
        
        
        
        

####################### select layers for training
train_generator =data.data_generator(train_dataset, model.config, shuffle=True,                                 
                        augmentation=augmentation,
                        batch_size=model.config.BATCH_SIZE,
                        no_augmentation_sources=no_augmentation_sources)

layers='heads'

# Pre-defined layer regular expressions
layer_regex = {
    # all layers but the backbone
    "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
    # From a specific Resnet stage and up
    "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
    "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
    "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
    # All layers
    "all": ".*",
}
if layers in layer_regex.keys():
    layers_sel = layer_regex[layers]
    

# learning_rate = config.LEARNING_RATE
# momentum =  config.LEARNING_MOMENTUM
# model.prep_train(learning_rate, momentum)

#from numba import cuda


# from tensorflow.contrib.memory_stats import BytesInUse
import tensorflow as tf
import tensorflow.keras.layers as KL
import gc
from time import sleep

epochs=5
learning_rate = config.LEARNING_RATE
momentum =  config.LEARNING_MOMENTUM
optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                    clipnorm=config.GRADIENT_CLIP_NORM)

losses=[]
for epoch in range(epochs):
    for step, input_train in enumerate(train_generator):
        
        # input_train = next(train_generator)
        
        if step==0 and epoch ==0:
            ypred = model(input_train[0])                  
            model.set_trainable(layers_sel)  
            mutil.load_weights(model.fpn.resnet,'./mask_rcnn_coco.h5',by_name = True)
           
        # else:
        #     #model = tf.keras.models.load('mymodel.h5')
        #     from mrcnn_tf2 import model as modellib
        #     model = modellib.MaskRCNN(mode="training", config=config,
        #                   model_dir=MODEL_DIR)
        #     ypred = model(input_train[0])
        #     sleep(0.1)
        #     model.load_weights('test.h5',by_name=True)
              
        #ypred = model(input_train[0])
        #print('step {}'.format(step))
        with tf.GradientTape() as tape:
            ypred = model(input_train[0])
            target_rois = tf.reduce_sum( tf.reduce_sum(tf.reduce_sum(model.target_rois, axis=2),axis=1),axis=0)
            nroi = target_rois.numpy()
            loss = tf.add_n(model.losses)            
        print('epoch:{}. step {}, nroi={}, loss={}'.format(epoch, step, nroi,loss))
        losses.append(float(loss.numpy()))
        if nroi>4:
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        tf.keras.backend.clear_session()
        # if step%5==0:
        #     model.save_weights('test.h5',overwrite=True)
        #     sleep(0.01)
        #     cuda.select_device(0)
        #     cuda.close()
        #     sleep(0.1)
        #     
        #     sleep(0.01)
        # tf.keras.backend.clear_session()
        # gc.collect()
        # del model, ypred
################

while True:
    input_train = next(train_generator)
    ypred = model(input_train[0])
    target_rois = tf.reduce_sum( tf.reduce_sum(tf.reduce_sum(model.target_rois, axis=2),axis=1),axis=0)
    nroi = target_rois.numpy()
    if nroi==0:
        break




model.compile(optimizer=optimizer)
        
          
############################3


active_class_ids =  mutil.parse_image_meta_graph(input_train[0][1])['active_class_ids']
ypred = model(input_train[0])
# outputs = [rpn_class_logits, rpn_class, rpn_bbox,mrcnn_class_logits, mrcnn_class,mrcnn_bbox]
rpn_class = ypred[1]
rpn_bbox = ypred[2]

##-----------------------------------------------
out_fpn = model.fpn(input_train[0][0])
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
rpn_rois = model.PL([rpn_class, rpn_bbox, anchors])
##### inner model.PL
deltas = rpn_bbox
scores = rpn_class[:,:,1]
###########

input_rpn_match = input_train[0][2]
input_rpn_bbox = input_train[0][3]
input_gt_class_ids = input_train[0][4]
input_gt_boxes = input_train[0][5]
input_gt_masks = input_train[0][6]
gt_boxes = mutil.norm_boxes_graph(input_gt_boxes, input_train[0][0].shape[1:3])
                
input_batch = [rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks]             
rois, target_class_ids, target_bbox, target_mask =model.DTL(input_batch)
############################

DTL = mutil.DetectionTargetLayer(config, name="proposal_targets_test")

proposals = rpn_rois[0]
gt_class_ids = input_gt_class_ids[0]
gt_boxes = gt_boxes[0]
gt_masks = input_gt_masks[0]








inputs_fpn =[rois, input_train[0][1], *mrcnn_feature_maps]    
# class, bbox        
mrcnn_class_logits, mrcnn_class, mrcnn_bbox = model.fpn_clf(inputs_fpn)

mrcnn_mask = model.fpn_mask(inputs_fpn)

x =mutil.PyramidROIAlign([14,14])(inputs_fpn)

# Conv layers
x = model.fpn_mask.mask_conv1(x)
x = model.fpn_mask.bn1(x, model.fpn_mask.train_bn)
x = KL.Activation('relu')(x)

x = model.fpn_mask.mask_conv2(x)
x = model.fpn_mask.bn2(x, model.fpn_mask.train_bn)
x = KL.Activation('relu')(x)

x = model.fpn_mask.mask_conv3(x)
x = model.fpn_mask.bn3(x, model.fpn_mask.train_bn)
x = KL.Activation('relu')(x)

x = model.fpn_mask.mask_conv4(x)
x = model.fpn_mask.bn4(x, model.fpn_mask.train_bn)
x = KL.Activation('relu')(x)

x = model.fpn_mask.deconv(x)
x = model.fpn_mask.mrcnn_mask(x)

from mrcnn_tf2 import model_loss as ML
# mrcnn_class_logits = ypred[3]
class_loss = ML.mrcnn_class_loss_graph(target_class_ids, mrcnn_class_logits, active_class_ids) 


# tf.saved_model.save(model,'rpn_only.h5')        
# model.save_weights('rpn_weights.h5')        
##################################
import tensorflow as tf
import tensorflow.keras.layers as KL
import multiprocessing
epochs=1
learning_rate = config.LEARNING_RATE
momentum =  config.LEARNING_MOMENTUM
opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                    clipnorm=config.GRADIENT_CLIP_NORM)

model.compile(optimizer=opt)

model.fit(x=train_generator )



workers = multiprocessing.cpu_count()
model.fit(
    train_generator,  
    epochs=epochs,
    steps_per_epoch=config.STEPS_PER_EPOCH,            
    validation_data=val_generator,
    validation_steps=config.VALIDATION_STEPS  
)


     

############################################33333
        
        
        
input_image = KL.Input(shape=config.IMAGE_SHAPE, name="input_image")

_,resnet = mutil.resnet_graph(input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)


from mrcnn_tf2.fpn_model import fpn_model
from mrcnn_tf2 import model_utils as mutil        
   
# input_image = KL.Input(shape= config.IMAGE_SHAPE)    
# r = mutil.resnet_graph(input_image, config.BACKBONE,stage5=True, train_bn=config.TRAIN_BN)

input_image = input_train[0][0] 
# out = r(input_image)

fpn = fpn_model(config, name='fpnn')

out = fpn(input_image)




