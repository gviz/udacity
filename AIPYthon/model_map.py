"""
    TorchVision pretrained model configuration
"""
from torchvision import datasets, transforms, models
model_map = { 
    "vgg16": {
    'model':models.vgg16,
    'ip_layer': [25088, 4096],
    'hidden_layer': [4096, 4096],
    'op_layer': [1000, 102]
    }, 
    "densenet121": {
        'model': models.densenet121,
        'ip_layer' : [1024, 1000],
        'hidden_layer' : [1000, 1000],
        'op_layer' : [1000, 102 ]
    }
    }