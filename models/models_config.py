import sys
import torch
from models.dcformer import decomp_tiny, decomp_naive, decomp_nano

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

class Config:
    def __init__(self, model, input_size, reduction=None):

        self.DIM_IMAGE = 12288
            
        if model == 'decomp_nano':
            patch_size = 64
            self.image_encoder = decomp_nano(input_size=[input_size,input_size,input_size//2])
            if reduction == 'channel':
                self.DIM_IMAGE = self.image_encoder.encoder.dims[-1]
            elif reduction == 'depth':
                self.DIM_IMAGE = self.image_encoder.encoder.dims[-1] * (input_size // patch_size) ** 2
            else:
                self.DIM_IMAGE = self.image_encoder.encoder.dims[-1] * (input_size // patch_size) ** 2 * (input_size // (patch_size * 2))
        elif model == 'decomp_naive':
            patch_size = 64
            self.image_encoder = decomp_naive(input_size=[input_size,input_size,input_size//2])
            if reduction == 'channel':
                self.DIM_IMAGE = self.image_encoder.encoder.dims[-1]
            elif reduction == 'depth':
                self.DIM_IMAGE = self.image_encoder.encoder.dims[-1] * (input_size // patch_size) ** 2
            else:
                self.DIM_IMAGE = self.image_encoder.encoder.dims[-1] * (input_size // patch_size) ** 2 * (input_size // (patch_size * 2))
            
        elif model == 'decomp_tiny':
            patch_size = 64
            self.image_encoder = decomp_tiny(input_size=[input_size,input_size,input_size//2])
            if reduction == 'channel':
                self.DIM_IMAGE = self.image_encoder.encoder.dims[-1]
            elif reduction == 'depth':
                self.DIM_IMAGE = self.image_encoder.encoder.dims[-1] * (input_size // patch_size) ** 2                
            else:
                self.DIM_IMAGE = self.image_encoder.encoder.dims[-1] * (input_size // patch_size) ** 2 * (input_size // (patch_size * 2))




