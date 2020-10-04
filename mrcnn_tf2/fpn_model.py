# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:20:05 2019

@author: Sang
"""



import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


# from mrcnn_tf2 import utils
# from mrcnn_tf2.model_utils import resnet_graph
from mrcnn_tf2 import model_utils as mutil
from mrcnn_tf2.resnet import resnet_graph





class fpn_model(KM.Model):
    def __init__(self, config, **kwargs):
        super(fpn_model, self).__init__(**kwargs) 
        self.config = config
        

        
        input_image = KL.Input(shape = tuple(config.IMAGE_SHAPE))
        self.resnet = resnet_graph(input_image, config.BACKBONE,
                                          stage5=True, train_bn=config.TRAIN_BN)
    
        # _, C2, C3, C4, C5 = resout
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        self.c5p5 =  KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')#(C5)
        self.p5up = KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")
        self.c4p4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')#(C4)
        # P4 = KL.Add(name="fpn_p4add")([
        #     KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        #     KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        
        self.p4up = KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")
        self.c3p3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')#(C3)
        # P3 = KL.Add(name="fpn_p3add")([
        #     KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        #     KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        
        self.p3up = KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")
        self.c2p2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')#(C2)
        # P2 = KL.Add(name="fpn_p2add")([
        #     KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        #     KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        self.P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")#(P2)
        self.P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")#(P3)
        self.P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")#(P4)
        self.P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")#(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        self.P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")#(P5)
        
    def call(self, inputs):
        
        _, C2, C3, C4, C5 = self.resnet(inputs)
        # Top-down Layers
        
        P5 = self.c5p5(C5)
        P4 = KL.Add(name="fpn_p4add")([
            self.p5up(P5),
            self.c4p4(C4)])
        

        P3 = KL.Add(name="fpn_p3add")([
            self.p4up (P4),
            self.c3p3(C3)])
        

        P2 = KL.Add(name="fpn_p2add")([
            self.p3up(P3),
            self.c2p2(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = self.P2(P2)
        P3 = self.P3(P3)
        P4 = self.P4(P4)
        P5 = self.P5(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = self.P6(P5)
        
        outputs =[P2, P3, P4, P5, P6]
        return outputs

    
    
    
    
    
    
# class fpn_model(KM.Model):
#     def __init__(self, config, **kwargs):
#         super(fpn_model, self).__init__(**kwargs) 
#         self.config = config    
#     def call(self, inputs):
#         config = self.config
    
#         _, C2, C3, C4, C5 = mutil.resnet_graph(inputs, config.BACKBONE,
#                                           stage5=True, train_bn=config.TRAIN_BN)
    
    
#         # Top-down Layers
#         # TODO: add assert to varify feature map sizes match what's in config
#         P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
#         P4 = KL.Add(name="fpn_p4add")([
#             KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
#             KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
#         P3 = KL.Add(name="fpn_p3add")([
#             KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
#             KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
#         P2 = KL.Add(name="fpn_p2add")([
#             KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
#             KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
#         # Attach 3x3 conv to all P layers to get the final feature maps.
#         P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
#         P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
#         P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
#         P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
#         # P6 is used for the 5th anchor scale in RPN. Generated by
#         # subsampling from P5 with stride of 2.
#         P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
        
#         outputs =[P2, P3, P4, P5, P6]
#         return outputs
    # return  KM.Model(inputs,outputs,name='fpn')
    
# # fpn = fpn_model(config)
# # ROOT_DIR = os.path.abspath("./")
# # COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# # fpn.load_weights(COCO_MODEL_PATH, by_name=True)

# # out = fpn(input_image)



       