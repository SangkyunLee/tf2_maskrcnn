# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 22:53:29 2019

@author: Sang
"""



import tensorflow as tf
# tf.keras.backend.clear_session()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], False)
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15*1024)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)



import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn_tf2 import utils
from mrcnn_tf2 import model_utils as mutil
from mrcnn_tf2 import model as modellib
from mrcnn_tf2 import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

from mrcnn_tf2.config import Config

###########################
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ShapesConfig()
config.display()


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes
    
    

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()


# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()



# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)



############################ training test -1

from mrcnn_tf2 import model_loss as ML
@tf.function
def compute_loss(model, inputs):
    # input_image = inputs[0]
    input_image_meta = inputs[1]
    input_rpn_match = inputs[5]
    input_rpn_bbox = inputs[6]
    
    # input_gt_class_ids = inputs[2]
    # input_gt_boxes = inputs[3]
    # input_gt_masks = inputs[4]
    
    config = model.get_config()
    
    out = model(inputs)
    
    rpn_class_logits = out[0]
    rpn_class = out[1]
    rpn_bbox = out[2]
    
    
    
    mrcnn_class_logits = out[3]
    mrcnn_class = out[4]  
    mrcnn_bbox = out[5]
    # 
    
    target_class_ids = out[6]
    target_bbox = out[7]
    target_mask = out[8]
    mrcnn_mask = out[9]
    

    
    rpn_class_loss  = ML.rpn_class_loss_graph(input_rpn_match, rpn_class_logits)
    rpn_bbox_loss = ML.rpn_bbox_loss_graph(config, input_rpn_bbox, input_rpn_match, rpn_bbox)
    
    active_class_ids =  mutil.parse_image_meta_graph(input_image_meta)['active_class_ids']
    class_loss = ML.mrcnn_class_loss_graph(target_class_ids, mrcnn_class_logits, active_class_ids)  
    
     
    bbox_loss = ML.mrcnn_bbox_loss_graph(target_bbox, target_class_ids, mrcnn_bbox)         
    mask_loss = ML.mrcnn_mask_loss_graph(target_mask, target_class_ids, mrcnn_mask)
    
    
    
    f=tf.keras.regularizers.l2
    weight = config.WEIGHT_DECAY
        
    reg_losses =[f(weight)(w)/tf.cast(tf.size(w),tf.float32)
                  for w in  model.trainable_weights
                  if 'gamma' not in w.name and 'beta' not in w.name]
    
    main_loss =  rpn_class_loss + rpn_bbox_loss +class_loss + bbox_loss + mask_loss 
    main_loss += tf.add_n(reg_losses)
    
    return main_loss        
    

@tf.function
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss
  



def train_generator():
    from mrcnn_tf2 import data
    image_index=0
    while True:
        inputs, outputs, image_index =data.gen_inputs(dataset_train, model.config, image_index, shuffle=True,                            
                        augmentation=None,                            
                        batch_size=model.config.BATCH_SIZE,                            
                        no_augmentation_sources=None)
        yield (inputs, outputs)
        if image_index>=len(dataset_train.image_ids)-1:
            break

output_types=((tf.float32, tf.float32, tf.int32, tf.float32, tf.int32, tf.float32, tf.bool),())    
dataset = tf.data.Dataset.from_generator(generator = train_generator,
                                      output_types = output_types) 

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
    



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Create model in training mode    
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

out = model.mrcnn(inputs)
# model.mrcnn.save('test1.h5', save_format='tf')
model.fpn.save_weights('test-fpn.h5')
model.fpn.load_weights('test-fpn.h5')

model.rpn.save_weights('test-rpn.h5')

# epochs=5
# learning_rate = config.LEARNING_RATE
# momentum =  config.LEARNING_MOMENTUM
# optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
#                                     clipnorm=config.GRADIENT_CLIP_NORM)

# losses=[]
# for epoch in range(epochs):
#     for step, inputs_ in enumerate(dataset):
        
#         # inputs_ = next(iter(dataset))
#         inputs = inputs_[0]
#         if step==0 and epoch ==0:
#             ypred = model(inputs)                  
#             model.set_trainable(layers_sel)  
#             mutil.load_weights(model.fpn.resnet,'./mask_rcnn_coco.h5',by_name = True)
        
#         loss = compute_apply_gradients(model, inputs, optimizer)
#         print('epoch:{}. step {},  loss={}'.format(epoch, step,loss))
#         losses.append(loss)



# model.save_weights('test.h5')

# model.save_weights('path_to_my_weights.h5')

# mutil.load_weights(model.fpn.resnet,'path_to_my_weights.h5',by_name = True)



# ################## DETECTION ###################
# class InferenceConfig(ShapesConfig):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1

# inference_config = InferenceConfig()

# # Recreate the model in inference mode
# model = modellib.MaskRCNN(mode="inference", 
#                           config=inference_config,
#                           model_dir=MODEL_DIR)


# # model(inputs)

# image_id = random.choice(dataset_val.image_ids)
# original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#     data.load_image_gt(dataset_val, inference_config, 
#                            image_id, use_mini_mask=False)
    

# inputs = [[original_image], image_meta]
# model.load_weights('path_to_my_weights.h5',by_name=True)
# mutil.load_weights(model.fpn.resnet,'path_to_my_weights.h5',by_name=True)




