"""
Mask R-CNN
The main Mask R-CNN keras-model implementation.

Written by Sangkyun Lee
This code is a modification of Waleed Abdulla from  Matterport, Inc.
This is built for tensorflow 2.0
Implementation of mask-rcnn to keras-model

"""

import os
import datetime
import re

# import logging
# from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf


import tensorflow.keras.layers as KL
#import keras.engine as KE
import tensorflow.keras.models as KM

from mrcnn_tf2 import utils
# from mrcnn_tf2.model_loss import *
from mrcnn_tf2 import model_loss as ML
#from mrcnn_tf2.model_utils import *
from mrcnn_tf2 import model_utils as mutil
from mrcnn_tf2 import data
from mrcnn_tf2.fpn_model import fpn_model


# Requires TensorFlow 2.0+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("2.0.0")

    
class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        
        
        assert mode in ['training', 'inference']
        

        
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self. counter=0
        
        
    # def build(self,input_shape=None):
        mode = self.getmode()
        config = self.get_config()
   
        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # feature pyramidal network model
        self.fpn = fpn_model( config, name='fpn_model')


        # # RPN Model             
        self.rpn = mutil.RPN(config.RPN_ANCHOR_STRIDE, 
                             len(config.RPN_ANCHOR_RATIOS),
                             name = 'rpn_model')
        
        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        self.PL = mutil.ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="Proposal",
            config=config)
       


        if mode == "training":
            #anchor generation
            self.get_anchors(config.IMAGE_SHAPE)

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            self.DTL =\
                mutil.DetectionTargetLayer(config, name="proposal_targets")
        else:
            self.DL = mutil.DetectionLayer(config, name="mrcnn_detection")
                
        

        # Network Heads        
        self.fpn_clf = mutil.fpn_classifier(config.POOL_SIZE, config.NUM_CLASSES, name='fpn_classifier')
        self.fpn_mask = mutil.fpn_mask(config.MASK_POOL_SIZE, config.NUM_CLASSES, name ='fpn_mask')
        
        
        self.mrcnn = self.build()
        
        # # Add multi-GPU support.
        # if config.GPU_COUNT > 1:
        #     from mrcnn.parallel_model import ParallelModel
        #     model = ParallelModel(model, config.GPU_COUNT)

    def get_config(self):
        return self.config
    def getmode(self):
        return self.mode
        
    def build(self):
        
        config = self.get_config()
        mode = self.getmode()
        
        
        
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE, name="input_image")
        
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        

        
        if mode == 'training':            
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            
            #input_rpn_match = inputs[5]
            #input_rpn_bbox = inputs[6]
        # else:
        #     input_anchors = inputs[2]
            
        # print('fpn_start')    
        out_fpn = self.fpn(input_image)
        # print('fpn_end')  
        rpn_feature_maps = out_fpn
        mrcnn_feature_maps = out_fpn[:-1]
                
          
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))
            
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))                   
                    for o, n in zip(outputs, output_names)]
        
        rpn_class_logits, rpn_class, rpn_bbox = outputs
        
        

        
        anchors = self.get_anchors(config.IMAGE_SHAPE)
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)    
        
        rpn_rois = self.PL([rpn_class, rpn_bbox, anchors])
        
            
        
        if mode == 'training':    
            # Normalize coordinates
            gt_boxes = mutil.norm_boxes_graph(
                input_gt_boxes, input_image.shape[1:3])
            
            # print('DTL_start')
            input_batch = [rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks]             
            rois, target_class_ids, target_bbox, target_mask =self.DTL(input_batch)
            self.target_rois = rois
            # print('DTL_end')
            
        else:
            rois = rpn_rois
        
        
        
            
    
        inputs_fpn =[rois, input_image_meta, *mrcnn_feature_maps]    
        # class, bbox        
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_clf(inputs_fpn)
        
        if not mode == 'training':
            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates            
            inputs_dl =[rois,  mrcnn_class, mrcnn_bbox, input_image_meta ]
            detections = self.DL(inputs_dl)
            
            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            
            inputs_fpn =[detection_boxes, input_image_meta, *mrcnn_feature_maps]  
        
        # mask
        mrcnn_mask = self.fpn_mask(inputs_fpn)
        
        # if mode == 'inference':
        #     input_dl = [ rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta]
        #     detections = self.DL(input_dl)
    
    
        if mode == 'training':
            inputs = [input_image, input_image_meta, input_gt_class_ids,
                      input_gt_boxes, input_gt_masks]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits, mrcnn_class, mrcnn_bbox,
                       target_class_ids, target_bbox, target_mask, mrcnn_mask]
            
          
        else:
            inputs = [input_image, input_image_meta]
            outputs = [detections, mrcnn_class, mrcnn_bbox, 
                       mrcnn_mask, rpn_rois, rpn_class, rpn_bbox]
        
        return KM.Model(inputs, outputs)
            
   
        
    
    
    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        
        config = self.get_config()
        mode = self.getmode()
        assert mode == "inference", "Create model in inference mode."
        assert len(
            images) == config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        # if verbose:
        #     utils.log("Processing {} images".format(len(images)))
        #     for image in images:
        #         utils.log("image", image)

        # # Mold inputs to format expected by the neural network
        # molded_images, image_metas, windows = self.mold_inputs(images)

        # # Validate image sizes
        # # All images in a batch MUST be of the same size
        # image_shape = molded_images[0].shape
        # for g in molded_images[1:]:
        #     assert g.shape == image_shape,\
        #         "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # # Anchors
        # anchors = self.get_anchors(image_shape)
        # # Duplicate across the batch dimension because Keras requires it
        # # TODO: can this be optimized to avoid duplicating the anchors?
        # anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)

        # if verbose:
        #     utils.log("molded_images", molded_images)
        #     utils.log("image_metas", image_metas)
        #     utils.log("anchors", anchors)
        # # Run object detection
        # detections, mrcnn_class, mrcnn_bbox, mrcnn_mask,rpn_rois, rpn_class, rpn_bbox =\
        #     self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # # Process detections
        # results = []
        # for i, image in enumerate(images):
        #     final_rois, final_class_ids, final_scores, final_masks =\
        #         self.unmold_detections(detections[i], mrcnn_mask[i],
        #                                image.shape, molded_images[i].shape,
        #                                windows[i])
        #     results.append({
        #         "rois": final_rois,
        #         "class_ids": final_class_ids,
        #         "scores": final_scores,
        #         "masks": final_masks,
        #     })
        # return results
    
    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks
    
    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        config = self.get_config()
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)
            molded_image = mutil.mold_image(molded_image, config)
            # Build image_meta
            image_meta = mutil.compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows
    
    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        config = self.get_config()
        
        backbone_shapes = mutil.compute_backbone_shapes(config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                config.RPN_ANCHOR_SCALES,
                config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                config.BACKBONE_STRIDES,
                config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]
    
    
    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        
        config = self.get_config()
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
    
 
         
    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            print("Selecting layers to train")

        keras_model = keras_model or self.mrcnn

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            #if layer.__class__.__name__ == 'Model':
            if isinstance(layer, KM.Model):
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                print("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

           
        
    # def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
    #     """Sets model layers as trainable if their names match
    #     the given regular expression.
    #     """
    #     # Print message on the first call (but not on recursive calls)
    #     if verbose > 0 and model is None:
    #         utils.log("Selecting layers to train")

    #     # keras_model = keras_model or self.keras_model

    #     # # In multi-GPU training, we wrap the model. Get layers
    #     # # of the inner model because they have the weights.
    #     # layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
    #     #     else keras_model.layers
    #     if not model:    
    #         layers = self.layers
    #     else:
    #         layers = model.layers

    #     for layer in layers:
    #         # Is the layer a model?
    #         #if layer.__class__.__name__ == 'Model':
    #         if isinstance(layer, KM.Model):
    #             print("In model: ", layer.name)
    #             self.set_trainable(
    #                 layer_regex, model=layer, indent=indent + 4)
    #             continue

    #         if not layer.weights:
    #             continue
    #         # Is it trainable?
    #         trainable = bool(re.fullmatch(layer_regex, layer.name))
    #         # Update layer. If layer is a container, update inner layer.
    #         if layer.__class__.__name__ == 'TimeDistributed':
    #             layer.layer.trainable = trainable
    #         else:
    #             layer.trainable = trainable
    #         # Print trainable layer names
    #         if trainable and verbose > 0:
    #             utils.log("{}{:20}   ({})".format(" " * indent, layer.name,
    #                                         layer.__class__.__name__))        
    
    
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.
                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        
        mode = self.getmode()
        config = self.get_config()
        assert mode == "training", "Create model in training mode."

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
            layers = layer_regex[layers]

        # Data generators
        train_generator = data.data_generator(train_dataset, config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data.data_generator(val_dataset, config, shuffle=True,
                                       batch_size=config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        utils.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        utils.log("Checkpoint Path: {}".format(self.checkpoint_path))
        utils.set_trainable(self, layers)
        self.prep_train(learning_rate, config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)            
        
